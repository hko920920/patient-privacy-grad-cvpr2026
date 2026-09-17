# Downstream 실행 기반·시간 profile: 성능 비교 전의 한정 실행

2026-09-17 · 큰 단계2 / 방향1. 예상45–75분. 사용자가 통합 명세를 검토하고 다음 profile 실행을 승인했다. **좋고 나쁨은 구현 연결과 실제 비용으로 판단하며, 이번에는 private 효용 성공 여부를 판단하지 않는다.**

[통합 명세](TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md)의 개발 비DP→개발 DP→전체 동결→expert final 순서를 유지한다. 이 profile은 새 학습 경로의 정합성과 계산량만 확인한다. 최종532명 및 reserved4213명의 pixel·model prediction은 접근하지 않는다. 제외를 검증하기 위한 기존 역할 metadata만 읽는다.

## 실행 전 고정한 범위

- 실자료 manifest: public813장/672명, private320장/80명, selection5097장/2027명, method-development5047장/2026명. 합계11,277장. 실제 파일 SHA·bytes·size·mode를 검사하고 전부 grayscale full-frame224 letterbox cache를 만든다. 새 개발 pixel 소비를 이 단계부터 기록한다.
- Profile 생성: fresh CPU latent4개. 기흉4cell 및 normal/effusion/cardiomegaly 각1cell을 네 방법에 대응시켜 **28장** 저장한다. 첫 base cell에 대해 W0·원상복구 재실행2회, private-only 동일 입력 재실행1회로 **총31 decode**다. 최대32 제한 안이며 본512장과 다른 profile bank다. 이 불균형한 작은 묶음으로 질환 성능을 비교하지 않는다.
- 기존 E4·CFG7.5·FP32·DDIM30을 사용한다. Public/pooled는 검산된 의료 head, private-only는 기존 저장 통계 분석에서 산출한 동일 기반모델의 W를 사용한다. 새 solver·head fitting은 없다. 기존 학습에 사용한 정확한 기흉 prompt의 conditioning을 재사용한다. 영상별 latent·conditioning·weight·source SHA와 전체 trajectory를 기록한다.
- Private-only는 기존 saved noisy-state witness와 online correction을 비교한다. W0의 모든 주요 trajectory 배열과 복구 결과, private-only 재실행은 exact equality를 요구한다. CFG correction은 conditional branch에만 더하고 실제 guided 변화7.5배를 검산한다.
- Classifier는 공식 ImageNet1K V1 ResNet18 checkpoint `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`다. 전체 fine-tuning, FP32, batch32, AdamW lr1e-4/weight decay1e-4. 이 lr은 profile용 고정값이며 본실험의 공개 calibration 선택 결과가 아니다.
- 7개 arm×7step×2회 = **총98 optimizer updates**. 모든 arm의 시작 weights·seed11이 같고 반복2회에서 입력·logit·loss·최종 state의 exact equality를 확인한다. 100step을 각각 두 번 실행하는 것이 아니다.
- Patient-uniform→image-uniform, source별 class balance8/8, 공통 real16 draw와 synthetic cell draw를 방법 간 공유한다. 본 실험에서는 동일 kernel을 선택된400/800step에 적용한다. 이번에는 AUROC/AP·개발 효용 gate를 계산하지 않는다.
- 비용 확인을 위해 두 개발군에서 hash 선택한128장에 R1 profile 모델의 forward만 실행한다. 점수는 감사용으로 보존하되 selection·성능 우위 판단에 사용하지 않는다. Final/test는 들어오지 않는다.

## 검증과 한계

생산 코드의 데이터 전처리·sampler·trainer·sampling 함수를 가져오지 않는 별도 검산기로 manifest/입력hash, 전체 저장 cache 행,64장의 독립 raw preprocessing, DDIM/CFG/residual 산술, PNG 변환,98step provenance/class balance, 손실 재계산, 실제 저장 model state 동일성을 확인한다. 별도의 UNet·VAE 추론으로28장 모두를 다시 생성하는 검사는 아니다.

허용오차는 실행 전에 계약에 결속한다. 기존 witness replay와 residual은 atol/rtol2e-6, classifier replay와 W0 복원은 exact다. 독립 DDIM 검산은 float32 항별 크기를 반영한64×machine-epsilon bound, guided correction도 기존 고정식의64×machine-epsilon bound를 쓴다. 실패 후 유리하게 tolerance를 바꾸지 않는다.

진행 조건: 모든 주요 정합 검증 통과, 누락/오염 파일 없음, finite trajectory/gradients, 작업 cap 준수, 실측 비용 보고. 이 조건은 private generation utility나 CVPR 기여의 성공 조건이 아니다. 실패는 처음부터 기록하고 구현 오류와 연구 효용 실패를 구분한다.

Profile이 통과해도 full512장과21개 classifier run을 이번 profile 완료라고 세지 않는다. 실제 생성/학습/전처리/validation forward 시간을 합쳐 본 패키지 ETA를 다시 계산한다.7step의 초기·warm 비용과800step의 장기 처리량 차이 때문에 범위로 보고한다.

## 본 결과가 유망할 때 남는 최종 비교 보강

사용자 검토의 세 항목을 final 전 개발 과제로 남긴다: 더 강하면서 의료적으로 합당한 real augmentation, ImageNet 기반 다른 classifier architecture, 더 넓은 non-P prompt mixture와 독립 생성 bank. 이번에 결과를 보기도 전에 arm을 늘리지는 않는다. Expert final을 여는 시점에는 이 비교를 수행했는지 또는 어떤 범위로 claim을 제한했는지 명시한다.

이전192장 pooled 효용 미통과·두 평가기 실패는 유지한다. Private 실자료 진단을 DP 방법이라 부르지 않고, 생성 label을 전문의 정답으로 취급하지 않는다.

## 모델 실행 전 입력 형식 정정

최초 준비에서는 독립 재계산본의1e-15 수준 차이를 확인해 public/pooled를 원 생산 weight로 고정했다. V1 파일 검사는 SHA의 대소문자 비교를 정정하고 실패본을 보존했다. V2에서11,277장 검사는 완료했으나, 일반 추출 row에 conditioning/noisy가 있다는 잘못된 가정으로 생성 입력 준비가 중단됐다. 새 모델 forward와 decode는 모두0이었다.

V3에서는 과거에 완전한 입력이 저장된 고정 train witness로 private-only offline–online 연결을 검사한다. 기흉 conditioning은 공개 backbone cache의 동일한 사전 prompt에서 가져온다. 나머지 세 조건도 같은 공개 cache와 exact인지 확인한다. Prompt·seed·모델·수치 허용오차·실행 cap은 바꾸지 않는다. V2 원 코드와 명세를 보존하고, 이미 검증한 실자료 cache는 hash로 결속하여 재사용한다. 기존 설명 중 “기흉 일반 추출 row에서 conditioning을 읽는다”는 구현 가정만 이 정정으로 대체한다.
