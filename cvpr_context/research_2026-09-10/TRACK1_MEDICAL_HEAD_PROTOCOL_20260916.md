# 의료 공개 backbone 위의 비DP head: 고정 생성 비교

2026-09-16, 큰단계2·방향1. 목표는 고정 공개 E4/CFG7.5 위에서 public-only head와 public+private pooled head의 **실제 생성 차이**를 확인하는 것이다. backbone·CFG·rank·lambda·seed를 결과에 맞춰 조정하지 않는다. 본 패키지는 개발용 적용성 screen이며 clinical quality, private 고유 정보, patient-DP 또는 CVPR 기여를 확증하지 않는다.

## 1. 모델과 자료

- 조건부 채택된 E4 SHA `7bc5ce421cc02091e8c3812676e9c27dc58330afce5e9384abd8edc9f61a3b91`, pinned SD2.1/VAE, CFG7.5, FP32, DDIM30 eta0,256px, 기존 null/conditional embedding 정책 유지.
- 기존 public32명64장, private-role fit80명320장, 재사용 development40명160장. 환자당 동일 질량, 환자 내부 이미지/8draw 동일 질량. pooled는32/112와80/112로 결합한다. 역할과 final calibration/test를 변경하지 않는다.
- 두 역할 모두 NIH 공개자료를 역할 분리한 모사다. private-role 자료를 추가한 효과와 실제 비공개 병원 자료만의 고유 가치는 다르다. public은환자당2장,fit/development는4장이라는 기존 관측량 차이도 유지·명시한다.
- 같은 conv_out 직전320-channel tap과 데이터 독립 Gaussian-QR P만 exact 재사용한다. feature·base prediction·residual·A/B/Q·W·witness는 새 모델에서 전부 계산한다.

## 2. CFG를 포함한 학습목표

conditional feature로 `phi64`를 구성하고 conditional correction을 `Delta=phi W`로 정의한다. 실제 guided base는 `g0=eps_u+7.5*(eps_c-eps_u)`다. 학습은 **guided residual** `r=eps_target-g0`와 feature `X=7.5*phi`를 사용한다.

`L(W)=patient_mean[spatial_mean sum_channels(r-XW)^2] + 0.001*||W||_F^2`.

즉 A=mean(X'X), B=mean(X'r), Q=mean||r||², W=(A+.001I)^(-1)B다. lambda는 조건부 가중치 W에 적용하며 결과를 보고 배율/감쇠를 바꾸지 않는다. 실제 sampling은 unconditional branch를 그대로 두고 conditional branch에만 Delta를 더한 후 표준 CFG를 계산한다. guided 출력에 전달되는 값은 수학적으로7.5Delta다.

이는 기존 CFG1/conditional-only MSE 실험과 목표가 다르므로 과거 MSE 수치와 직접 비교하지 않는다. 강한 guidance를 실제-noise MSE로 보정하면 유용한 조건 반응을 약화할 가능성도 있다. 목표를 맞춘 것이 생성 효용을 보장하지 않으며 실제 생성 비교로 판단한다. DP·새 solver·비선형 head·추가scale 탐색은 없다.

## 3. 추출·정합 검산

총544장×8draw=4,352기록, 두 branch batched UNet 한 번/기록, warmup1, backward0. 기존 image/draw별 noise와 8개 timestep stratum을 유지한다. public/train/eval의 첫 영상 draw0/7은 hidden/noisy/conditioning witness도 저장한다. 처음8기록 실측을 비용 profile로 보고하고 loss로 조정하지 않는다.

새 raw feature·두 branch epsilon·guided base·target을 저장한다. 환자 통계, 가중치와 개발 MSE는 독립 순서/Cholesky로 다시 계산한다. witness에서 투영·basis·noise 생성·분기 순서와 runtime inference를 비교한다. 원 모델/adapter 불변과 gradient 부재를 확인한다.

생성 전에 W=0의 base 30-step 복원, offline/online correction, `eps_guided=eps_u+7.5*((eps_c+Delta)-eps_u)`, hook 제거 후 원상복구를 검사한다. 수학적 `g-g0=7.5Delta`의 FP32 연산 순서 차이는 사전에 고정한 오차 경계로 검산한다. 구현 검산 실패는 품질 실패로 해석하지 않고 원인을 보존·정정한다. 생성에서 NaN/Inf가 나면 확대를 멈추고 기록한다.

## 4. 실제 생성과 평가

세 방법 `backbone`, `public`, `pooled`만 비교한다. salt `medical-head-generation-20260916-v1|seed|index`의 SHA256 앞15hex로 정한 새16개 CPU latent를 네 고정 prompt가 공유한다. 기존 개발/확인 seed와 중복을 검사한다. **방법당64장, 총192장**, 같은64 cell끼리 paired 비교. 이미지 수와 seed는 결과를 보고 늘리지 않는다. 독립 noise block은16개다.

전부 저장하고 opaque-ID 비교 grid를 먼저 판독한다. 큰 형태 오류와 public/pooled 간 눈에 보이는 변화는 비임상 관찰로 기록하며, 그림이 좋아 보인다는 이유로 유리한 seed만 고르지 않는다.

기존 pinned RAD-DINO·BioViL-T 평가기 원리를 재사용한다. 기존448명 전부는 사용하지 않는다. 기존 모든 evaluation 역할·auxiliary 역할·backbone 학습 환자를 제외한 reference 중 no-finding/effusion을 각20명씩 고정 hash 순서로 선택한 **40명**을 사용한다. 실제 명부 확인에서 모든 auxiliary 역할까지 제외하면 두 조건이30명/20명 남으므로, 최초 초안의36명씩은 실행 전에20명씩으로 정정했다. 모델 결과를 보고 바꾼 것이 아니다. 환자 중복과 실파일 hash를 먼저 검증한다. 과거 evaluator 개발에 사용된 reference이며 새 final test가 아니다.

- reference가 대응하는 normal/effusion 생성32장에 대해 raw RAD-DINO full-sample unbiased KID와 PRDC k5를 계산한다. 두 조건이 각각16장/20명으로 균형을 이룬다. n32이므로 과거 KID subset50 정책을 호출하지 않는다.
- generic/cardio를 이 reference에 억지로 합치지 않는다. 전체64장의 feature diversity는 별도 기술 통계로 기록한다.
- BioViL-T는 동일 prompt의 cosine 및 세 specific prompt 사이의 matched-minus-other cosine을 paired 비교한다. generic은 alignment 성공 주장에 넣지 않는다. weak prompt와 encoder 한계가 있으므로 임상 질환 정확도로 부르지 않는다.
- seed block별 pooled-public 차이와 descriptive bootstrap 범위를 기록한다. 반복 bootstrap은 새로운 생성표본이 아니다. 검증된 최소 임상 효과크기나 모집단 유의성으로 해석하지 않는다.
- 원 이미지, feature 배열, prompt vectors와 per-image 점수를 저장하고 metric을 별도 산술로 검산한다. encoder의 고정된 일부 입력 재실행으로 수치 반복성을 확인한다.

## 5. 판정과 범위

생성 효용 판단이 primary이며 MSE는 보조다. pooled가 public보다 여러 생성 지표와 대응 관찰에서 의미 있게 나아지는지, 차이가 수치 반복 오차 수준인지, 품질/정렬/다양성 사이의 tradeoff인지 구분한다. 서로 엇갈리는 지표를 임의로 골라 성공이라고 하지 않는다. 사적 자료의 추가 생성 가치가 불분명하거나 악화하면 새 DP solver 실행을 보류한다.

본192장은 단일 head/split의 개발 screen이다. 좋은 방향이라도 독립 환자·조건/seed·public budget과 강한 DP 비교는 후속이다. 새로운 확증 실험이나 DP는 이 패키지에 자동 포함하지 않는다.

전체 예상45–75분. 추출과 생성 각각 실측으로 갱신한다. 실행 소스는 단계별 실행 이전에 고정하며, 최초 데이터/목표/예산은 본 문서로 먼저 고정한다. 첫 추출부터 결과를 보고 방법·기준·참조를 바꾸지 않는다. 이전 동결 코드/실패/채택 기록은 수정하지 않는다.
