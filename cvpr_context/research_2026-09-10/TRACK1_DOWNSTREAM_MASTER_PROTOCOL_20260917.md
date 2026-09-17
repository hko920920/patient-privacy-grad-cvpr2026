# 기흉 downstream utility 통합 계획: 개발에서 비DP·DP를 끝내고 expert final은 마지막에 평가

2026-09-17 · 큰 단계2 / 방향1. **설계와 자료 기반에는 긍정적이다. 방법 성능은 아직 미실행·미판정이다.** 사용자 검토를 반영해 ‘비DP를 expert final에서 확인한 뒤 DP를 설계’하는 순서를 금지한다. 개발자료에서 실행 가능성, 비DP 효용, DP 구성과 비용을 검토한 뒤, 최종 비교군·모델·분석을 함께 동결한다. 그 이후에만 전문가 final 성능을 연다.

이 문서는 단계별 의사결정과 초기 개발 recipe를 정한 **master protocol v1**이다. 아직 downstream runner, classifier checkpoint 결속, 새 의료 head의 DP 구현·accounting과 DP-LoRA profile이 없다. 따라서 최종 실행 계약이 완성됐거나 지금 expert final을 실행할 수 있다는 뜻은 아니다. `final_ready=false`다.

[직전 expert 환자 감사](NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md) · [전문가 라벨 출처](NIH_EXPERT_LABEL_PUBLIC_COPY_20260917.md) · [기존 192장 결과](TRACK1_MEDICAL_HEAD_RESULTS_20260916.md) · [측정 주장 정정](MEASUREMENT_CLAIM_REVIEW_20260917.md)

## 1. 검증할 주장과 아직 주장할 수 없는 것

질문: **공개 의료 생성모델의 작은 적응층에 사적 역할 환자자료를 더하면, 그 합성자료가 실제 기흉 분류의 효용을 높이는가? 그 효용을 환자 단위 DP 아래 유지하면서 patient-DP LoRA보다 유리한 전체 비용을 만들 수 있는가?**

주요 효용은 합성자료를 추가해 학습한 분류기의 실영상 성능이다. 기흉 label 0은 ‘기흉 음성’이며 ‘정상’이 아니다. 생성영상의 개별 병변 정확성, 임상 사용 적합성, 사적 자료만의 고유 정보는 이 결과만으로 입증되지 않는다. NIH 공개자료의 public/private 역할 모사이며 실제 비공개 병원 코호트라는 주장도 하지 않는다.

기흉을 새 prospective primary task로 둔다. 폐기종7명은 primary나 방법 선택에 쓰지 않는다. 기흉 분류기의 출력으로 폐기종 AUC를 계산하지도 않는다. 별도 폐기종 출력을 사전에 학습·동결하지 않는 한, 이번 연구의 폐기종 결과는 자료 규모 설명뿐이다. 이전 폐기종+기흉 공동 자동평가기 실패는 그대로다.

Downstream 평가는 기존 DP-LoRA에서도 사용한 효용 축이다. 다만 그 논문은 합성자료 학습/실제자료 test를 사용했고, 여기서는 공개 실자료에 합성자료를 추가하는 증강 질문을 택한다. 동일한 평가를 재현했다고 쓰지 않는다. [DP-LoRA 원논문 §4, 로컬 B08 본문 확인](https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.html)

## 2. 이번에 실제 계산한 자료 구성

명부·실제 backbone training trace·로컬 acquisition inventory만 읽었다. 새 image pixel, classifier/generator forward, 품질 점수는 읽거나 계산하지 않았다.

|역할|환자|영상|기흉 양성 영상|기흉 양성 환자|용도|
|---|---:|---:|---:|---:|---|
|기존 공개 backbone640 + public head32의 실제 입력 합집합|672|813|6|6|모든 downstream arm의 공통 실제 학습자료|
|기존 private head 자료|80|320|33|17|생성기 적응; 별도 비DP 실자료 진단 arm|
|Classifier selection 후보|2,027|5,097|230|114|공개 실자료 기준 분류기 학습 설정 선택|
|Method development 후보|2,026|5,047|223|114|비DP 및 이후 DP의 개발 비교|
|별도 기존 reserved confirmation|4,213|이번에 계산 안 함|—|—|그대로 보존; 자동 사용 안 함|
|Expert final|532|810|136|86|모든 최종 모델과 분석을 동결한 뒤 평가|

공개813장은 이미 public으로 사용한749+64장이다. 새 개발 환자를 공개 학습자료에 대량 편입해 private 정보의 필요성을 바꾸지 않는다. 여기에 기흉 양성이6장뿐이라는 사실은 실제 기존 구성의 한계다. Private80에는33장이 있으므로 추가 task 정보의 가능성은 합당하지만, 작은 공개 양성군의 과적합 위험도 크다.

남은 개발4,053명은 이전 감사의 개발4,213명에서 두 evaluator 신규80명씩을 제외한 집합이다. 환자별 관측 기흉 양성 유무로 층을 나눈 뒤 고정 salt와 patient-ID SHA 순서로 교대 배정했다. 각 영상의 원래 weak label은 바꾸지 않았다. 두 새 역할, 공개672, private80, 기존 dev40, reserved4213, expert532의 쌍별 교집합은 모두0이다.

명부는 `code_working/_reports/downstream_master_20260917_v1`에 별도 prospective overlay로 저장했다. 원 졸논 역할·원 사용 이력·예약 집합은 수정하지 않았다. 새2,027/2,026명의 pixel이나 성능을 아직 소비하지 않았다. 이 자료는 보호대상 private80과 분리된 연구 개발 보조자료이며, 원래 역할명이 thesis private_train이었다는 기록도 유지한다.

## 3. 개발과 final의 순서

|단계|수행|판정 자료|다음 단계의 조건|
|---|---|---|---|
|A. 실행 기반|아래 runner·명부 결속·작은 시간 profile|공개813 및 개발자료만|라벨·sampling·학습·통계 경로가 맞고 예상 비용이 확인됨|
|B. 비DP 가능성|고정7개 arm, 공통 classifier3 seeds|Method development2,026명|Private/pooled 합성자료의 추가 효용 후보가 있음|
|C. DP 및 강한 비교|공개 보정된 same-head DP-SGD와 patient-DP LoRA; 필요시 근거 있는 기존 public-assisted 구성|개발자료만|효용·비용·privacy 구현이 검증되고 비교 구성이 완성됨|
|D. 최종 동결|비DP·DP 전체 arm, 모든 weights/data/seed/code/accounting/statistics manifest 저장|Final 성능 미열람|미정 항목0, 실패 arm 포함 여부도 결과 전에 결정|
|E. 최종 평가|동결한 모든 arm을 expert810장에서 평가|Expert532명|사전 분석 그대로 보고; 결과 후 방법 변경 없음|

‘한 번’은 GPU 프로세스 하나만 허용한다는 뜻이 아니라 **하나의 사전 동결된 최종 평가 라운드**다. 모델별 추론을 순차 수행해도 결과를 가리고 전체가 끝난 뒤 분석한다. 중간에 일부 결과를 보고 다른 arm의 모델·학습·추론을 바꾸지 않는다. 중단 후 동일 weights와 입력을 재개하는 기술적 복구는 기록하고 허용한다. 의미적 버그 수정이나 결과 후 새 arm은 별도 탐색으로 표시하며 같은 final의 독립성을 새로 주장하지 않는다.

기존 졸논에서 이 expert 환자의 성능을 먼저 열어 연구 선택에 사용해도 독립성 문제가 생긴다. 두 연구가 같은 final 사용 이력을 공유한다. 공식 metadata·표본 수를 확인한 것과 모델 성능을 열어 본 것은 구분한다.

## 4. 첫 비DP 개발 recipe

### 4.1 생성기와 합성자료

- 기존 공개 E4 LoRA SHA `7bc5ce421cc02091e8c3812676e9c27dc58330afce5e9384abd8edc9f61a3b91`, CFG7.5, FP32, DDIM30, eta0, 256px를 유지한다.
- 방법은 backbone, public head, private-only head, pooled head의4개다. 모두 새 의료 backbone에서 계산한 full64 가중치를 사용하고 원 weight SHA를 계약에 넣는다. Generic-base용 head·과거 M1/M2는 사용하지 않는다.
- 초기 개발은 방법당128장, 총512장이다. 기흉 요청64장, 비기흉 요청64장으로 맞춘다. 후자는 기존 normal22, effusion21, cardiomegaly21 prompt를 고정 혼합한다. 음성 합성자료를 전부 정상으로 만들지 않는다.
- 기흉 prompt는 `a frontal posteroanterior chest radiograph with radiographic findings of pneumothorax` 하나다. 나머지는 기존 medical head 생성 계약의 literal prompt를 그대로 복사·hash 결속한다. 기존 학습/생성 label 템플릿과의 일치도 실행 전에 확인한다.
- 새 초기 latent64개를 CPU에서 만들어 저장한다. 각 index는 기흉 요청과 고정된 음성 prompt 하나에 사용하고, 네 방법 모두 정확히 같은 latent tensor를 공유한다. 과거 생성·확인 seed와 중복 여부도 실행 전 검사한다. 결과를 보고 prompt·seed·CFG·head scale을 변경하지 않는다.
- 기흉0/1 합성 label은 **요청한 조건**이다. 전문의 정답이 아니며 동반 기흉이나 잘못된 질환 표현이 없다고 가정하지 않는다. 이 label noise까지 포함한 합성자료의 실제 학습 효용을 묻는다. 자동평가기 점수로 좋은 영상만 거르지 않는다.
- 과거512장과 독립인 두 번째 생성 bank 없이 latent 모집단 일반화를 확정하지 않는다. 큰 실험의 생성 수·bank 수는 개발 결과와 실제 비용으로 정하되 expert final을 열기 전에 동결한다.

### 4.2 비교군: 같은 계산량 안에서 비교

|Arm|학습 입력|질문|
|---|---|---|
|R0|공개813, 기본 전처리만|실자료 기준|
|R1|공개813 + 고정 일반 augmentation|일반 증강만으로 가능한 효과|
|S0|R1 실자료 + backbone synthetic|공개 생성기 자체의 효과|
|S1|R1 실자료 + public-head synthetic|공개 head의 추가 효과|
|S2|R1 실자료 + private-only synthetic|사적 역할 자료를 쓴 합성의 추가 효과|
|S3|R1 실자료 + pooled synthetic|기존 결합 방식의 추가 효과|
|Dreal|공개 실자료 + private 실자료|Private80의 실자료 추가가 이 분류 과제에 도움이 되는지; 비DP 진단용|

S2−S1과 S3−S1을 둘 다 보고한다. S1보다 좋더라도 R1보다 나쁘면 유용한 증강 방법이라고 부르지 않는다. Dreal은 경험적 비교점이며 효용의 수학적 상한도, 보호 방법도 아니다.

총 optimizer step과 batch size를 동일하게 고정한다. 합성자료를 추가했다고 epoch당 step을 늘리지 않는다. Batch32: 합성 arm은 real16(양성8/음성8)+synthetic16(8/8), R0/R1은 real32(16/16), Dreal은 public16+private16이며 각각8/8이다. 결과를 보고 클래스 비율을 바꾸지 않는다. 영상 추출은 해당 class에 영상이 있는 환자를 균일 추출한 다음 그 환자의 해당 class 영상을 균일 추출한다. 반복 방문 환자가 영상 수 때문에 학습을 지배하지 않게 한다. Public6 양성의 반복 사용량은 반드시 보고한다.

**Real-repeat를 별도 성능 arm으로 부풀리지 않는다.** 이처럼 균일 반복 sampling과 총 step을 고정하면 데이터를 단순 복제한 real-repeat는 R1과 같은 sampling 분포다. R1이 반복·일반 증강 대조 역할을 한다. 동일 draw stream에서 물리적 CSV 복제 여부가 학습 입력을 바꾸지 않는지 작은 구현 검사를 한다. 유한 epoch 방식으로 바꾸면 이 등가성이 깨지므로 재명세가 필요하다.

### 4.3 Classifier와 선택 범위

- 첫 후보는 **ImageNet-1K V1 ResNet18 전체 fine-tuning, binary logit1개**다. NIH 의료 사전학습 evaluator를 그대로 가져와 final 독립성을 주장하지 않는다. 정확한 torchvision 버전·공식 weight SHA·전처리는 runner 계약에 결속한다. 현재 로컬 ResNet18 checkpoint는 아직 확보 전이다. [공식 architecture/weight source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html)
- Grayscale를3채널로 복제하고 종횡비를 보존해 전체 영상을224 안에 넣은 뒤 검은 padding과 ImageNet normalization을 공통으로 사용한다. 기흉 소견이 있을 수 있는 주변부를 잘라내지 않기 위해 center crop을 피한다. 이는 공식 ImageNet 평가 transform의 그대로 재현이 아니라 명시적인 과제용 전처리다. R1과모든S/D arm의 augmentation은 rotation±5도, translation±2%, scale .95–1.05의 작은 affine만 사용한다. 좌우 반전·질환별 조작은 없다. Validation/test는 deterministic preprocessing이다. Resize interpolation·padding 위치·전처리 tensor parity를 실행 계약에 결속한다.
- AdamW, weight decay1e-4, BCEWithLogitsLoss, balanced batches, batch32. Public R1에서만 lr `{1e-4,3e-4}` × checkpoint `{400,800 steps}`를 selection2,027명으로 비교한다. 두 lr run을800step까지 돌리며400/800을 저장한다. 공통 calibration seed 하나, highest AUROC를 택하되 차이 .002 이내면 적은 step, 그다음 작은 lr 우선이다. 이는 개발 선택 규칙이지 임상적 차이 기준이 아니다.
- 선택된 lr/step을 모든7개 arm과 사전 seed `{11,23,37}`에 동일 적용한다. Arm별 early stopping·private/pooled만 추가 tuning은 없다. 초기 classifier weights와 real draw/augmentation stream도 가능한 공통 부분에서 대응시킨다.
- 이 classifier는 고정 외부 모델에 합성영상을 판정시키는 ‘세 번째 질환 평가기’가 아니다. 서로 다른 학습자료로 훈련되는 downstream 과제 모델이다. 원 expert final은 이 선택에 사용하지 않는다.

## 5. 비DP 개발 결과의 해석과 중단선

실제 개발 성능은 image label로 AUROC/AP를 계산하고 환자 군집을 고려한다. Seed3개 각각과 seed 평균의 차이를 모두 남긴다. 이 pilot에서 환자 모집단·생성 seed·classifier 변동을 모두 정밀 추정했다고 하지 않는다.

초기 운영 기준: S2 또는 S3가 S1과 R1보다 평균 AUROC에서 모두 **0.01 이상** 높고, 각각 최소2/3 classifier seeds에서 같은 방향이며, AP 평균을 낮추지 않으면 DP 개발에 투자할 후보로 본다. 0.01은 이번 소규모 실험의 투자 기준이며 임상적 최소 차이나 통계적 검정력 보장이 아니다. Expert final 성공 기준으로 재사용하지 않는다.

- Private 실자료도 개선되지 않으면 task 정보·작은 공개 양성군·classifier 학습의 문제를 구분해 해석한다. Private 정보가 원래 없다고 단정하지 않는다.
- Dreal은 좋아지지만 S2/S3는 좋아지지 않으면, 현재 생성/작은 head가 task-relevant 정보를 전달하지 못한 후보 결과다. 새 DP solver가 해결한다고 가정하지 않는다.
- 작은 양의 차이지만 위 투자 기준 미달이면 ‘미결정/효과 작음’으로 기록한다. 같은 seed·prompt·threshold를 바꿔 성공으로 만들거나 final을 열지 않는다. 추가 비용을 들일 근거가 있는지 보고한 뒤 새 개발 명세를 정한다.
- 계산이나 학습이 실패하면 방법 성능 음성 결과와 구별한다. 원인·수정·전체 재실행 범위를 기록한다.

## 6. DP 단계까지 이어지는 비교와 보호 경계

비DP가 유망하면 새 의료 CFG7.5 head의 DP 구성을 개발자료에서 검증한다. 기존 generic-SD2.1 DP 결과·가중치·clip/scales를 이식하지 않는다. 새 backbone은 fixed public E4이고 projection이 자료 독립임을 확인한다.

**기본 비교:** 공개 초기화와 공개 gradient를 이용하는 same-head patient-DP-SGD를 경량 구성의 대표로 둔다. 더 좋은 기존 public-assisted solver가 있다면 그대로 활용할 수 있다. Solver 신규성은 필수 주장이 아니다. 같은 solver를 ‘제안 방법’과 다른 이름의 baseline으로 두 번 세지 않는다. Public geometry의 독자 이득을 주장할 때만 별도의 구현·비교가 필요하다.

**강한 비용 비교:** 같은 public E4 기반의 patient-DP LoRA를 함께 둔다. 새로운 private adapter만 학습하고 환자당 여러 영상의 gradient를 모은 뒤 환자 단위로 clipping한다. Image-DP ε를 patient-DP ε와 같다고 쓰지 않는다. 공식 DP-LoRA의 모듈/학습과 의료 patient 확장의 차이를 기록한다. LoRA의 public-only calibration과 non-DP 동작 대조도 있어야 한다. 이 구현·GPU memory/time profile은 아직 미완료다.

Primary 보호 예산은 ε8, δ1e-5; 낮은 예산 ε4는 사전 secondary 후보로 둔다. N0=80 고정 공적 분모, add/remove-one-patient 인접성으로 정한다. 예산별 noise multiplier, sampling probability, step 수, clipping bound는 의료 head의 공개자료 calibration 및 독립 accountant 대조 후 실행 계약에 결속한다. 최종 DP recipe가 없는 지금 ‘accounting 완료’라고 쓰지 않는다.

초기 same-head 목적은 기존 pooled와 같은 public32/private80 환자 질량이다. Private per-patient gradient 평균→clip→Gaussian noise→고정 qN0 분모를 사용하고 private 질량80/112를 적용한다. Public 질량32/112와 ridge gradient는 별도로 더한다. Empty sampled batch도 정해진 mechanism step으로 처리한다. Private-only 목적이나 public-centered penalty를 추가하면 다른 사전 arm이며 pooled와 이름만 바꾼 동일 문제인지 확인한다.

**반드시 막을 누출:** 비DP private W로 초기화, private feature로 projection 선택, 비보호 private gradient norm으로 clip 선택, private 성능으로 가장 좋은 noise seed를 공개하는 경로. Public-only calibration과 고정 recipe를 우선 사용한다. Private-dependent selection을 mechanism 안에 넣으면 그 선택까지 별도 privacy accounting이 필요하다.

연구용 NIH 공개자료에서 이미 실행한 비DP 분석 전체가 나중에 DP로 바뀌지는 않는다. 보장은 동결한 training/release mechanism의 사적 역할80명에 관한 것이며, 탐색 기록·raw cache·비DP control까지 포함한 전체 artifact의 end-to-end DP 보장이 아니다. 실제 비공개 코호트로 배포하려면 연구 선택과 별개로 고정 recipe를 적용하거나 선택 절차까지 보호해야 한다.

DP generator 한 개의 합성자료와 공개자료로 학습하는 classifier는 추가 private access가 없을 때 post-processing이다. 하지만 여러 DP generators/예산/반복 중 선택하거나 함께 공개하는 비용은 단일 ε라고 뭉개지 않는다. 개별 모델 보장과 공동 공개의 composition을 구분한다. DP noise seed·raw private 통계를 공개하면 이 보호 설명을 유지할 수 없다. [DP post-processing 및 composition 원전 §2.3/3.5](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

## 7. Expert final의 고정 분석

Final manifest가 완성될 때 아래 순서를 적용한다.

1. Primary는810장/532명 전체의 **원래 영상별 expert pneumothorax label**이다. AUROC primary, average precision 방식 AUPRC secondary. 기흉0의674장은 정상군이 아니다.
2. 반복 방문은 모두 유지한다. 환자 any-positive로 모든 방문 label을 바꾸지 않는다. Primary는 영상 가중 estimand임을 명시하고, patient-equal weighted AUROC/AP는 secondary sensitivity로 둔다.
3. 환자532명을 replacement로 뽑고 해당 환자의 모든 영상을 multiplicity만큼 포함하는 **paired cluster bootstrap**을10,000회 사용한다. 같은 draw를 모든 arm/seed에 공유한다. 한 class만 남는 draw는 invalid로 기록하고 사전 규칙에 따라 재추출한다.
4. 먼저 classifier seed별 metric을 구한 뒤 평균한다. Seed별 예측을 먼저 평균하면 ensemble 성능으로 다른 estimand가 되므로 혼동하지 않는다. 환자 bootstrap CI는 저장된 훈련 모델에 조건부임을 명시한다. 모든 seed별 차이와, 최종 manifest에서 고정한 생성 bank/DP training seed 변동을 별도로 보고한다. 3개 classifier seeds만으로 학습 변동의 정밀 추정을 주장하지 않는다.
5. 비DP 두 비교 S2−S1, S3−S1을 둘 다 보고하고, 다중성에 보수적인 동시 구간으로 각97.5% 양측 cluster percentile CI를 제시한다. 둘 중 final에서 좋은 것만 primary로 고르지 않는다. 약한 개발자료에서 선택한 DP 대표·primary contrast도 final 전에 기록한다.
6. DP 주장은 선택·동결된 경량 DP 구성의 ε8 결과가 S1 위에 추가 효용을 유지하는지와, 같은 patient-DP LoRA에 대한 효용·전체 비용을 함께 본다. Primary superiority contrast와 family 크기는 최종 manifest에 열거하고 그 수에 따라 다중성 기준을 고정한다. Non-inferiority를 주장한다면 허용 margin과 근거도 final 전 별도로 고정해야 한다. 현재는 임의 margin을 임상적 허용치로 선언하지 않는다.
7. 전체810장 결과가 primary, 기존 final_test466명/741장만의 결과가 미리 정한 sensitivity다. 두 분석 중 좋은 쪽을 골라 headline으로 쓰지 않는다. 필요시 환자당 한 장 hash 선택도 secondary로만 두며, 이미지 수와 양성 수가 달라짐을 표시한다.
8. Threshold metric은 별도 비잠금 개발자료에서 고정한 threshold가 있을 때만 보고한다. Final에서 sensitivity/specificity를 좋게 만드는 threshold를 선택하지 않는다. AP는 해당810장 표본의 prevalence에 대한 결과이며 일반 모집단 값이라고 쓰지 않는다.

86명의 양성 환자는 평가 기반이지만1–2% AUROC 차이 검출의 보장은 아니다. Final 예측을 보지 않고 개발 prediction을 이용해 군집·arm 상관에 따른 불확실성 simulation을 할 수 있다. 이것도 expert label에서의 power를 보장하지 않는다. CI가 넓으면 불확실한 결과를 그대로 남긴다.

## 8. 비용과 다음 실행 범위

총 비용은 public backbone 사전학습(공유 비용을 별도 표시), feature 추출, DP/private adaptation, 합성 생성, downstream classifier 학습을 분리해 측정하고 합계도 보고한다. Private backbone forward/backward, peak allocated/reserved VRAM, wall time, 생성당 overhead를 기록한다. Same-head DP-SGD도 backbone backward0이므로 이를 새 solver만의 이득으로 쓰지 않는다.

기존 의료 head256px 생성은 약2.7초/장이었다. 이 수치의 단순 외삽으로512장은 **약23분의 생성 계산만** 필요하다. 모델 로딩·파일 IO·분류기 학습·검산은 별도이며 새 prompt batch의 실측이 아니다. 기존 SSP .015초 대 SGD .534초를 전체 속도 우위로 쓰지 않는다.

다음 한 작업은 **runner와 명부 결속,32장 이하 생성 및 classifier100step 이하의 시간/정합 profile**, 예상45–75분이다. 비용 profile의 이미지는 별도 개발 bank로 표시하고 본512장과 분리한다. 아직 이 작업을 실행하지 않았다. Profile 결과로 첫 비DP 패키지의 총 시간을 다시 보고한 뒤 진행한다. 현재 예상은 구현 이후 계산·검산·분석을 합쳐 추가60–120분이며 분류기 IO/VRAM에 따라 바뀔 수 있다. DP-LoRA의 시간은 아직 측정 근거가 없어 제시하지 않는다.

최종 계약의 미완료 항목: downstream runner/weights hash, 고정 literal prompt의 runner 계약 결속, fresh latent manifest, 공개 R1 calibration 결과, 새 의료 DP clipping/accounting, LoRA 구현·profile, 최종 합성 수·생성 bank/DP seed 수, 전체 arm 및 통계 family. 음성 prompt 세 개는 이미 JSON 계획에 복사했고 기존 medical-head 계약과 문자열 일치도 확인했다. 나머지는 개발 단계에서 닫을 목록이며 expert final을 먼저 열 이유가 아니다.

## 9. 이번 산출물의 범위

완료: 통합 연구 순서, 첫 개발 비교 recipe, 실제 자료 규모 계산, 고정 해시 기반 prospective 개발 명부, 보호·통계·최종 동결 규칙. Expert final 모델 성능은 미열람이다.

검증: 생산 inventory 함수를 가져오지 않은 pandas 재집계가 환자·영상·양성 수와 hash 배정에 일치했다. Public/private/두 개발군/expert 사이10개 쌍의 환자·image ID·기록된 파일 SHA 교집합은 모두0이다. 이는 inventory에 저장된 SHA 기준이며 이번에 원본 영상을 다시 해시·디코딩한 것은 아니다. 입력6개와 과거 결과 보고서4개의 불변, 두 상태 파일의 실제 결과 포인터 유지도 확인했다. [기계 판독 계획](spec_sources/downstream_master_plan_20260917.json) · [검증 기록](spec_sources/downstream_master_verification_20260917.json). 검증 PASS는 설계·명부 무결성이며 효용이나 검정력이 아니다.

미완료: 합성512장, downstream classifier 학습, private 효용, 새 의료 patient-DP, LoRA 비교, expert final, CVPR 기여.

원 source와 명부는 그대로다. 새 계획으로 과거 CheXzero/PadChest 실패나 pooled192장 효용 미통과를 성공으로 바꾸지 않았다. 공식 전문가 CSV와 가공본의 행 단위 동일성 미확인도 유지한다. 설계가 좋아졌다는 평가와 방법이 좋다는 평가는 계속 구분한다.
