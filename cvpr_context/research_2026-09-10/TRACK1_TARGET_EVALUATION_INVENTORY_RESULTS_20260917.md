# 폐기종·기흉 평가자료 점검: 현재 배정으로는 진행 불가, 평가기 후보는 확보

2026-09-17 KST. **판정: 자료 확보에는 부정적, 조건 민감 평가기 확보에는 긍정적이다.** 현재 고정된 자료 배정으로 두 질환의 새 생성 효용을 판단할 충분한 환자 reference를 만들 수 없다. 이는 private 신호가 무효라는 실험 결과가 아니라, 후속 가설을 검증할 자료가 부족하다는 결과다. 기존192장 pooled 추가 효용 실패, public backbone 채택 범위, DP 보류를 유지한다.

[실행 범위](TRACK1_TARGET_EVALUATION_INVENTORY_PROTOCOL_20260917.md) · [직전 신호 분석](TRACK1_PRIVATE_SIGNAL_RESULTS_20260917.md) · [기존192장 생성 비교](TRACK1_MEDICAL_HEAD_RESULTS_20260916.md)

## 1. 실제로 확인한 것

전체 local NIH **42,423장·14,755명**의 inventory를 공식112,120장 metadata와 대조했다. 기존905명 public-backbone 명부만 다시 센 것이 아니다. 원본 분할, 공식 train/test 목록, 현재 CVPR 역할, 과거 M1/M2 학습, evaluator preflight448명, 생성 reference40명, 공개 backbone 및 기록된 진단 이력을 함께 확인했다.

현재 학습·CVPR 역할과 분리된 public 후보 **424명·511장 전부**의 파일 존재, 크기, SHA256와 PNG decode를 확인했다. 원 acquisition hash 및 source-audit hash와 모두 같았고 기존 기록상 타 환자 exact-byte/pixel duplicate는0이었다. 새 perceptual near-duplicate 검색이나 임상 영상 판독은 아니다. 잠긴 calibration/test·official test/census 영상은 이번 점검에서 열지 않았다.

CPU inventory **11.470초**, 별도 원본 metadata 중심 재집계·파일 검증 **19.926초,2,124항목 PASS**. 이 숫자는 표본 수나 통계적 확증이 아니다. 새 모델 추론·GPU·생성·학습·DP는 모두0이다.

## 2. 환자 수가 실제로 얼마나 남는가

아래 숫자는 해당 label이 붙은 **서로 다른 환자 수**다. Multi-label 환자는 여러 조건에 포함될 수 있다.

|범위|전체 환자 / 영상|폐기종|기흉|No Finding|Effusion|
|---|---:|---:|---:|---:|---:|
|전체 public_development|1,816 / 4,831|61|127|1,265|404|
|공개 backbone640 + 모든 CVPR evaluation/auxiliary752명 제외|424 / 511|**3**|**6**|223|54|
|추가로 과거 reference·기록된 진단 환자 제외|264 / 264|**3**|**1**|172|18|
|그중 기존 예약 역할도 없는 환자|0 / 0|0|0|0|0|
|위424명 중 과거 evaluator/reference에 사용됨|159 / 244|0|5|51|36|

현재 head32/80/40명과 잠긴 calibration140/test140명은 CVPR752명 제외 집합 안에 있다. M1/M2 각각456명의 학습 환자도 이 집합의 부분집합임을 별도로 검산했다. 이미지가 달라도 같은 환자는 제외했다.

마지막0명은 **전 세계에 자료가 없다는 뜻이 아니라 현재 배정표의 빈칸이 없다는 뜻**이다. 하지만 예약 역할을 새 명세에서 재검토한다고 해도 질환별 최대치가3명/1명이며, 과거 평가 자료 재사용까지 허용한 느슨한 범위도3명/6명뿐이다. 따라서 이번 결론이 단지 엄격한 역할 명칭 때문인 것은 아니다.

공개 backbone nontraining 예약은 selection128/confirmation128/reserve9명이다. 각각 폐기종1/1/1명, 기흉0/1/0명이다. Reserve의 진단 이력1명·3장을 추가로 제외하면265명·267장에서264명·264장이 된다. 이 역할을 바꾸거나 환자를 새 reference로 채택하지 않았다. ‘미사용’은 조사한 기록상 결과 노출이 없다는 의미이며, acquisition/source-audit 이력과 같은 NIH 출처라는 한계는 남는다.

조건당20명은 이전 파일럿과의 실무적 비교 규모였으며 임상적·통계적 합격선이 아니다. **3명/1명으로 KID 수치를 만들 수 있다는 사실과 private 생성 이득을 신뢰성 있게 비교할 수 있다는 것은 다르다.** 현재는 후자에 부족하다.

## 3. 더 받으면 바로 해결되는가

공식 train_val PA metadata 중 기존 환자 배정 전체와 공식 test 환자를 제외하면 **14,113명·22,329장**이 추가로 존재하지만, 로컬 파일은0장이다.

|기존 배정 밖·미다운로드 후보|환자|영상|
|---|---:|---:|
|폐기종|157|180|
|기흉|**0**|**0**|
|No Finding|11,866|17,692|
|Effusion|0|0|

기존 NIH cohort가 기흉·흉수 등 target 환자를 모두 포함하도록 구성됐기 때문에, 배정 밖에서 기흉 환자가 새로 나타나지 않는다. 따라서 **기존 배정을 유지하면서 NIH 파일만 추가로 받는 방식은 두 target과 control을 갖춘 이번 비교를 해결하지 못한다.** 폐기종만 남겨 원래 두 조건 실험의 성공으로 처리하지 않는다.

자료 자체는 다른 기존 역할에도 있다. 예를 들어 original private_train에는 폐기종273명398장, 기흉563명1,065장이 있다. 그러나 이는 현재 졸논 분할의 훈련 역할이며, 이번에 새 확인자료로 전환하거나 과거 미학습 환자라고 인증한 것이 아니다. Official test/census와 privacy_attack_holdout도 이번 개발을 위한 저장고로 사용하지 않는다. Census-only는 이미지 분류이므로 같은 환자가 final_test에도 존재할 수 있고 두 표의 환자 수를 합산하면 안 된다.

## 4. 조건 민감 평가기는 시작할 수 있는가

**후보 파일은 이미 있다. 실행 검증은 아직 없다.** 설치된 TorchXRayVision1.3.5의 소스와 cached checkpoint5개를 읽고 hash를 고정했다. 공식 문서는 모든18개 출력이 모든 checkpoint에서 학습된 것은 아니며, 해당 모델의 유효 라벨만 써야 한다고 명시한다. [공식 모델 안내](https://github.com/mlmed/torchxrayvision#models)

|고정 가중치 후보|폐기종 출력 학습|기흉 출력 학습|이번 NIH 평가에서의 판단|
|---|---|---|---|
|PadChest-only DenseNet121 (`pc`)|있음|있음|우선 후보. 선언된 학습 출처에 NIH 없음|
|NIH DenseNet121|있음|있음|NIH 학습 중복 가능성 때문에 독립 평가기라고 주장 불가|
|다중자료 `all`|있음|있음|NIH 포함. 보조로만 검토|
|MIMIC-CXR `mimic_ch`|**없음**|있음|폐기종 슬롯을 유효 점수로 사용 불가|
|CheXpert `chex`|**없음**|있음|두 target 공동 평가기로 부족|

표의 trained label은 설치된 모델 소스의 실제 label 목록을 파싱해 확인했고 [공식 소스](https://github.com/mlmed/torchxrayvision/blob/main/torchxrayvision/models.py)와 대조했다. PadChest-only 파일은28,381,804bytes, SHA256 `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`이다. NIH와 다른 출처라는 것은 출처 수준 판단이며 모든 데이터 계보의 독립성을 증명한 것은 아니다.

No Finding 전용 출력은 위 모델들에 없다. 정상 control은 실제 정상 reference와 조건별 점수의 비교로 정의해야 하며 빈 출력 index로 만들면 안 된다. 입력 grayscale·범위[-1024,1024]·center crop·224resize와 출력 sigmoid/operating-threshold 변환 규칙을 후속 명세에 고정해야 한다. 점수를 환자의 보정된 질환 확률로 해석하지 않는다.

실제 가중치 로드, 전처리 replay, NIH 양성/음성 환자 구별력, 생성 artifact에 대한 반응은 이번에 실행하지 않았다. 그러므로 상태는 **checkpoint/label 준비 완료, 해당 목적에서의 evaluator validation 미완료**다. 이후 실제 환자 자료에서 조건별 AUC/AP와 환자 단위 불확실성을 확인하고, raw matched cosine과 함께 오르는 비특이적 점수를 제외할 근거가 있어야 한다. 모델을 생성 결과가 유리한 순서로 골라서는 안 된다.

RAD-DINO KID/PRDC와 BioViL-T는 기존에 실행됐지만, 새 두 조건에서의 판별력이 자동으로 확인되는 것은 아니다. 특히 RAD-DINO의 NIH 사전학습 중복과 BioViL raw cosine의 비특이성은 유지한다.

## 5. 진행 결정과 현실적인 다음 한 단계

**현재 고정 데이터 배정에서 targeted non-DP 생성과 DP 확대는 진행하지 않는다.** 이번에는 새 solver나 head를 만들어 해결할 병목이 아니다. 데이터가 부족해서 가설을 아직 평가하지 못한 상태와, targeted 생성 실험을 해봤더니 실패한 상태를 구별한다.

다음은 **새 데이터 구성의 타당성 검토20–30분** 한 단계다. 우선 별도 CVPR 연구에서 기존 final/test와 현재 head/backbone 환자를 보존하면서, 아직 사용하지 않은 NIH 훈련 역할 환자를 새 target-reference로 명시적으로 분리할 수 있는지 실제 학습·사용 이력을 조사한다. 이번에는 그렇게 전환하지 않았다. 별도 연구 분할을 만들 경우 기존 졸논 분할과의 관계 및 최종 시험의 독립성을 새 명세로 고정해야 한다. 이것은 신호가 유리한 환자를 고르는 것이 아니라, 두 target 및 controls의 전체 적격 후보를 정해 hash로 배정하는 작업이어야 한다.

그 경로가 타당하지 않으면 외부 자료의 조건 라벨·사용 조건·patient ID·PA view를 확인하는 대안으로 간다. 외부기관 reference만 가져와 NIH private adaptation 성공이라고 주장하면 acquisition shift와 cohort 차이가 섞일 수 있으므로 target train/reference의 의미도 함께 재설계한다. 외부 자료가 PadChest라면 현재 PadChest 학습 분류기의 overlap도 다시 검토해야 한다. 외부 자료를 확보했다고 보고하거나 다운로드하지 않았다.

자료 구성과 평가기 검증을 통과할 때에만 새 명세에서 backbone/public/private-only/pooled, 폐기종/기흉 및 normal/effusion, fresh shared latents를 고정한다. 조건 특이성 개선과 일반 품질 손실을 함께 보고, 결과를 본 뒤 질환·seed·기준을 바꾸지 않는다. 이 순서는 기존192장 실패를 성공으로 다시 이름 붙이는 절차가 아니다.

## 6. 재현 기록과 한계

실행 폴더: `code_working/_reports/target_evaluation_inventory_20260917_v2`. 전체 patient ID·외부 미다운로드 metadata 후보 목록은 local-only JSON에 있다. source/contract, inventory, file checks, independent verification, 최종 decision을 분리한다.

- Inventory SHA256: `09cfaa992e33c735db85083ffa62d1a68686c9047eeaa32e7d62ffdf89be65dd`.
- Contract SHA256: `73c51eb68c617852affcca923fc33297d7df3b22fed0bfed7970e1007f6d68e6`.
- 독립 검산은 생산 집계 함수를 가져오지 않고 공식 metadata로 환자·라벨 집합을 다시 만든다. 후보511장 SHA를 다시 읽었고 기존192장 생성물도 그대로임을 확인했다.
- 최초v1은32개 영상의 label 문자열 순서가 달라 중단됐다. 예: `Mass|Atelectasis`와 `Atelectasis|Mass`. Label 집합 차이는0이었다. 원 실행 contract/실패 source/실패 기록을 보존한 뒤, v2는 전체 label 집합·patient ID·PA view를 검사했다. 환자 선택이나 결과 허용 기준을 완화하지 않았다.
- 소수 metadata만으로 새 분포나 private 고유 의미 정보를 증명하지 않았다. 근접 중복·임상 라벨 오류·실제 분류 성능은 별도 문제다.
- 로컬 실행/파일 검증 결과이며, 이 작업에서 원격main SHA 확인이나 push를 했다고 주장하지 않는다.

**현재 위치: 큰 단계2·방향1. 사적 추가 방향의 실마리는 유지되지만, 현재 데이터 배정에서 검증을 확대할 준비는 부족하다.**
