# 방향1: PadChest 평가기 실영상 검증 — 실행 전 명세

2026-09-17. 큰 단계2·방향1. 예상20–35분. 목적은 고정 평가기가 실제 NIH 영상에서 폐기종·기흉을 구별하는지 확인하는 것이다. 새 생성·학습·DP는 하지 않는다. [직전 사용 이력 감사](TRACK1_PATIENT_USAGE_AUDIT_RESULTS_20260917.md)의 후보 분할과 기존 실패 결과를 유지한다.

## 1. 점수 확인 전 자료 선택

기존 비잠금 evaluator pool324명의 보유 PA 영상 label을 환자 단위로 합치면, 서로 배타적인 폐기종4명·기흉34명·두 target 모두11명·target 없는 흉수85명·target/흉수 없는 No Finding95명·기타95명이다. 앞선15/45명은 dual-positive를 포함했다. 폐기종 단독4명으로는 충분한 검증이 어려우므로 **추론 전에** 새 target-development에서 네 primary군 각각20명, 총80명을 evaluator-only로 분리한다. Reserved confirmation은 사용하지 않는다.

- 환자의 현재 보유 PA 영상 label 합집합으로 E/P 포함 여부를 정한다. 이는 모든 생애 방문에서 질환이 없다는 뜻이 아니다. E-only/P-only도 다른 질환 동반을 허용한다.
- 순서: 두 target 모두 → E-only → P-only → 두 target 없는 Effusion → 앞의 세 질환이 없는 환자의 No Finding 영상 → 기타. 환자군은 상호배타적이다.
- 기존 pool의 primary218명과 dual11명은 모두 사용하고 기타95명은 제외한다. 기존군은 이미 소비된 공개자료이므로 evaluator 개발용 보조 결과다.
- 별도 development80명이 이번 **주 판별력 검증군**이다. Salt `cvpr-padchest-validation-20260917-v1`; SHA256(salt+'|patient|'+정수환자ID) 순서로 각 군 첫20명을 고른다. 결과 기반 보충은 없다.
- 환자당 해당군 label을 가진 영상1장, SHA256(salt+'|image|'+imageID) 최솟값. No Finding은 단독 label이다. Dual 환자는 양 target 동시 label 영상이 있으면 그 안에서 고르고, 없으면 어느 target이든 포함한 영상 중 고른다. Dual 결과는 선택 영상 label로만 secondary 분석하며 두 독립 표본으로 세지 않는다.
- 총309명·309장. 개발 후보 E-only84명 중20명을 분리하므로 생성 reference 후보64명이 남는다. 나머지 군도20명씩 제거한다. 기존 졸논 역할과 final/test는 변경하지 않는다. 새 evaluator80명의 ID는 이후 생성 reference에서 반드시 제외한다.
- 선택 명부·source SHA·실행 코드·본 명세를 contract에 결속한 뒤 추론한다. 실제 선택 파일의 SHA/PNG decode/환자/label을 다시 검증한다. 전체 local inventory에서 cross-patient exact duplicate를 조사하고 있으면 중단한다. Near-duplicate 임상 영상 유사성 판정은 이 패키지에서 하지 않는다.

## 2. 단일 평가기·전처리·출력

TorchXRayVision1.3.5 `densenet121-res224-pc`만 사용한다. PC checkpoint SHA256 `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`. NIH/all 및 다른 모델 점수를 보고 교체하지 않는다. [공식 모델 설명](https://github.com/mlmed/torchxrayvision#models)과 설치된 models.py/datasets.py/utils.py의 hash를 기록한다. PC의 Emphysema/Pneumothorax/Effusion 출력은 실제 학습된 slot이다.

8-bit PNG를 grayscale로 해석하고 `(2*(pixel/255)-1)*1024` → XRayCenterCrop → XRayResizer224(skimage engine) → float32/단일채널. 모든 처리의 소스와 버전을 기록한다. GPU FP32/eval/inference mode, TF32 off, 고정 seed, batch16. 학습/역전파0.

주 점수는 `model.classifier(model.features2(x))`의 **raw logit**이다. 기본 `model(x)`는 sigmoid 후 checkpoint operating-threshold 정규화를 하므로 raw logit이나 NIH에서 보정된 확률로 부르지 않는다. 저장된 logits와 실제 default forward 출력을 4개 고정 replay 입력에서 비교하고, sigmoid/op_norm을 독립 계산한다. FP32 batch 차이 허용치는 raw logit 절대2e-4/상대2e-5, 변환 산술은 절대2e-6으로 미리 고정한다. 미학습 slot은 분석하지 않는다. 질환 간 logit 절대 크기가 교정돼 있다는 가정도 하지 않는다.

## 3. 사전 고정 분석·운영 기준

각 stratum을 섞지 않고 E/P 각각 다음을 계산한다: target 대 나머지 세 primary군, target 대 상대 target, target 대 No Finding, target 대 target-free Effusion. AUC, AP와 표본 prevalence, 양음성 score분포를 기록한다. 예측 threshold는 정하지 않는다. 2,000회 환자 bootstrap은 각 군 내에서 복원추출하며 군 크기를 유지한다. 95% percentile 구간, 고정 seed `20260917`. 환자당1장이므로 영상=환자 단위이며, dual은 별도 기술 통계만 보고한다.

이번 진행 기준은 임상 인증·검정력 보장이 아니라 **생성 비교에 쓸 보조 평가기의 개발 관문**이다. 주 validation80명에서 두 target 각각:

1. 대 나머지군 AUC ≥0.70이며 bootstrap95% 하한 >0.50.
2. 대 상대 target AUC ≥0.60이며 bootstrap95% 하한 >0.50.

조건부 통과 여부를 각 target과 joint로 모두 보고한다. 다중 metric의 비보정 구간이며 최종 유의성 주장으로 쓰지 않는다. AP는 case-control 구성의 prevalence에 의존하므로 두 stratum AP를 일반 population AP처럼 직접 비교하지 않는다. 두 target 중 하나만 통과하면 전체 평가기 검증 완료라고 쓰지 않는다. 실패/불명확 시 기준·환자·모델을 사후 바꾸거나 confirmation으로 보충하지 않는다.

Real-image 판별력 통과도 generated artifact에 대한 강건성이나 임상적 타당성을 보장하지 않는다. 후속 생성 비교에서 일반 품질·조건특이성·artifact/diversity를 함께 봐야 한다. 다른 질환 score와의 단순 raw-logit 차이는 교정 없이 primary로 쓰지 않는다.

## 4. 저장·검산·분기

출력 `code_working/_reports/padchest_validation_20260917_v1`. 선택 명부, evaluator 제외환자, 잔여 개발 후보, source contract, 정규화 tensor, 전체 logits, 실제 replay 출력, scalar metric/bootstrap, 실행 시간·GPU 메모리와 결과를 저장한다. 기존192장·기존 실패 adoption·원래 분할은 불변 확인한다. 검산기는 생산 metric 함수를 import하지 않고 환자 선택/중복/전처리 산술/저장 출력/ROC pair-count AUC 및 AP를 재계산한다. 별도 full model 재학습 또는 임상 독립판독으로 표현하지 않는다.

통과하면 다음 단계는 **별도 고정 targeted 비DP 생성 비교 명세**다. 이번 작업이 private 효용을 입증하거나 DP 실행을 허가하지 않는다. 미통과하면 부족한 조건과 원인을 먼저 보고하고 현재 evaluator로 전체 private utility 성공을 판정하지 않는다. 결과 보고에는 좋음/나쁨/불명확과 다음 작업 예상 시간을 명시한다.
