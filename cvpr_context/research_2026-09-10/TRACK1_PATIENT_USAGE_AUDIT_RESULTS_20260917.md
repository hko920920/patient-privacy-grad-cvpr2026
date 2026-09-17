# 실제 사용 이력 감사: 별도 CVPR 확인자료를 구성할 근거 확보

2026-09-17 KST, 큰 단계2·방향1. **판정은 자료 구성에 긍정적이다.** 기존 public_development 안에서는 폐기종3명·기흉1명뿐이었지만, original private_train의 실제 사용 이력을 확인하니 별도 연구용 후보 **8,426명**이 남았다. 새 reference를 만들 수 없다는 보편적 결론은 맞지 않다. 이전 inventory가 확인한 ‘현재 공개 역할 안에서는 부족하다’는 결과는 그대로 유효하다.

**이 결과는 private 생성 효용이나 patient-DP 성공을 뜻하지 않는다.** 평가 가능한 자료 구성을 마련할 근거다. 이번에는 역할 원본을 변경하거나 새 모델·분류기 추론, 생성, 학습, DP를 실행하지 않았다.

[감사 규칙과 사전 hash 분할](TRACK1_PATIENT_USAGE_AUDIT_PROTOCOL_20260917.md) · [직전 공개자료 inventory](TRACK1_TARGET_EVALUATION_INVENTORY_RESULTS_20260917.md) · [기존192장 실제 생성 결과](TRACK1_MEDICAL_HEAD_RESULTS_20260916.md)

## 1. 역할 이름과 실제 소비를 어떻게 구분했는가

원래 졸논 NIH `private_train`은8,476명이다. 하지만 본학습 완료를 의미하지 않는다. 실제 실행된 K5 시험 학습은3개 arm 각각4step이고, 원본 `restricted_runtime_diagnostics.json`에 환자와 영상 ID가 남아 있다. 이 기록을 공개 report의 SHA와 대조했다.

- 실제 사용: **50명·59장**. M1 두 arm은 같은37개 image-unit schedule을 공유하고 M2는13개 patient-unit을 처리했다. 중복을 제거한 환자/영상 수를 사용했다.
- 그50명의 다른 이미지까지 환자 단위로 제외하므로 보유175장이 빠졌다.59장만 빼고 같은 환자의 다른 영상을 확인자료로 쓰지 않았다.
- K5 full-run gate에는 본학습 미시작이 명시돼 있고, 현재 지정된 `_restricted_runs` 및 full-run arms 경로도 없다. 실행 source는 selected runtime units만 VAE/text 준비 후 학습한다. 전체 manifest를 읽었다는 사실을 전체 환자의 모델 소비로 세지 않았다.
- 기존 CVPR M1/M2는 각각456명·912장에 대해 실제 coverage 완료 보고와 검산이 있다. 이들은 original public_development 출신으로 후보와 환자가 겹치지 않는다.
- 공개 E4/E8 backbone은 실제 committed trace에서749장·640명을 재구성했다. Head32/80명, 개발40명, reference40명, evaluator preflight 및 CVPR cache/공격 기록도 따로 표시했다.

추가 누락 탐지를 위해 NIH/CVPR 관련 **3,680개 JSON·CSV·JSONL,182,820,591bytes**를 조사해 patient/image ID 참조를 확인했다. 선언·metadata/intake와 모델 결과/사용 가능성 기록을 구분했다. Original private_train에서는 위50명 이외의 추가 사용 hit가 없었다.

이것은 **조사한 로컬 실행 기록상 미사용**이라는 판단이다. 외부 경로에서 미기록 실행이 전혀 없었다는 증명은 아니다. 관련 없는 ISIC/합성 fixture ID를 NIH 환자로 오인하지 않았고, source/file inventory 읽기만으로 전 환자를 학습했다고 표시하지 않았다. 조사 파일 목록과 hash를 모두 저장했다.

## 2. 환자별 이력표를 실제 만들었다

Local 전체14,755명에 대해 `patient_usage_private.csv`를 생성했다. 다음 열이 들어 있다.

`patient_id`, `original_thesis_role`, backbone/public-head/private-head/개발loss/reference/evaluator/시각진단/선택·가설 flags, M1/M2 및 졸논 dry-run/기타 disposable 학습, attack/model diagnostic, feature cache, 미해결사용·exactduplicate flags, 잠긴 졸논/현재 CVPR 여부, 질환별 보유영상 수, 개발/확인 후보 여부, 근거 파일 수.

`patient_evidence_private.json`에는 각 사용 flag의 근거 파일을 연결했다. Flag0은 조사 범위에서 해당 증거가 발견되지 않았다는 뜻이다. 시각 판독을 모델 feature 진단과 동일시하지 않았다. 최종 등급은 복수 flag를 보존한 채 E→A→B→C→U→D 순서로 표시한다.

|등급|의미|환자 수|
|---|---|---:|
|A|실제 학습에 소비됨. 폐기된 시험 모델의 학습도 포함|1,112|
|B|학습 A/E 외, 개발loss·선택·가설에 소비됨|24|
|C|A/B/E 외, 평가·진단·cache에 소비됨|186|
|D|조사 기록상 모델/결과 소비 없음|10,506|
|E|original official test/census 및 CVPR locked cal/test|2,927|
|U|추가 미해결 사용 가능성 또는 exact duplicate로 별도 보류|0|

E등급은 실제 학습 flag가 있어도 우선 보존한다. 이 표는 상호배타적 요약이고, 예를 들어 ‘A1,112명’이 모든 과거 학습 환자의 총합이라는 뜻은 아니다. U0도 외부 미기록 실행의 부재를 증명하지 않는다.

D10,506명 중 **8,426명은 original private_train**,1,816명은 원래 privacy_attack_holdout,264명은 public의 기존 예약 역할이다. 이번 별도 CVPR 후보는 **original private_train의8,426명으로만 한정**했다. 다른 잠긴/예약 역할을 숫자를 늘리기 위해 가져오지 않았다.

## 3. 두 target과 control이 함께 충분히 남는가

|원래 훈련 역할 중 실제 사용 이력 제외 후보|환자 수|해당 label 영상 수|
|---|---:|---:|
|전체|8,426|21,581|
|폐기종|**273**|398|
|기흉|**559**|1,058|
|No Finding 영상 보유|5,726|10,728|
|두 target과 겹치지 않는 Effusion 영상 보유|1,780|2,822|

폐기종·기흉 두 label의 영상을 가진 환자 교집합은110명이다. Label 수를 합쳐 독립 환자 수라고 부르지 않는다. 정상 control은 영상의 `No Finding` 단독 label, 흉수 control은 영상에 두 target label이 없는 경우로 명세를 고정했다. 다른 방문에서 같은 환자가 다른 조건을 가질 수 있으므로 환자 군집을 보존해야 한다.

환자별 label 수는 공식 metadata와 실제 local acquisition inventory로 재계산했다. 현재 후보 명부에서 선택될 영상들의 파일 존재·bytes는 확인했으나, 이번에는 새 영상의 pixel을 열거나 전체21,581장을 다시 해시하지 않았다. 이후 실제 사용하는 작은 evaluator/reference 명부를 확정할 때 acquisition SHA와 실제 파일 hash를 결속해야 한다.

## 4. 사전 hash 규칙으로 별도 확인자료가 실제 가능한지 계산했다

감사 명세에 먼저 고정한 salt와 patient ID의 SHA256 순서로 전체 후보를 반씩 나눴다. 질환별로 유리한 환자나 눈으로 좋아 보이는 영상을 선택하지 않았다. 두 split 모두4,213명이다.

|가상 연구 역할|폐기종 환자|기흉 환자|정상 control 환자|흉수 control 환자|
|---|---:|---:|---:|---:|
|Target development/reference 후보|**135**|**268**|2,845|905|
|별도 reserved confirmation 후보|**138**|**291**|2,881|875|

둘 사이 환자 교집합0, 현재 backbone/head/private80/development40·M1/M2·CVPR 기존 역할·official test/census와 교집합0이다. 각 split의 폐기종/기흉 환자 교집합은51명/59명이고 조건별 표본 독립성을 가정하지 않는다.

조건별 환자당1장을 동일 salt의 image ID hash로 고르는 가상 명부도 만들었다. 총8,338 condition rows이며 한 환자 또는 영상이 여러 label 행에 나타날 수 있다. **8,338개 독립 환자나 당장 전부 평가할 영상 수가 아니다.** 전체 환자 분할과 일관된 영상 선택 가능성을 점검한 것이다.

명부는 `provisional_patient_split_private.csv`와 `provisional_condition_images_private.csv`로 저장했다. 이것은 새 실행에서 채택할 수 있는 구체적 후보이며 **기존 졸논 분할 변경이나 final reference 채택이 아니다.** 생성 수, reference sample 수, 조건 간 가중치, 성공 기준과 power는 별도 실행 명세에서 정해야 한다. Reserved confirmation은 그 선택 과정에 열지 않는다.

## 5. 졸논과 충돌하지 않게 사용할 수 있는 조건

같은 원자료가 연구별로 다른 역할을 가질 수 있다. 현재 CVPR 모델은 이 후보8,426명으로 학습하지 않았으므로 별도 CVPR reference로 고려할 수 있다. Original 졸논 명부는 그대로 남겼다.

그러나 **향후 졸논 K5/K10 모델이 이 환자를 훈련에 사용하면, 그 모델에 대해서는 같은 명부가 독립 reference가 아니다.** 따라서 후속 CVPR baseline도 현재 private adaptation cohort와 새 reference 분리를 지켜야 한다. 전체 졸논 훈련모델을 가져와 같은 reference에서 비교하는 것은 허용되지 않는다. 새 실행 계약은 평가 대상 모델의 실제 train patient set과 reference set을 다시 대조해야 한다.

현재 private80을 바꾸거나 target에 유리한 새 train cohort를 만들지 않았다. 이번 다음 실험이 답할 질문은 그대로다.

> 기존 분석에서 발견한 폐기종·기흉 조건에서, 고정된 private-only/pooled head가 public-only보다 실제 생성 효용을 추가하는가? 이를 가설 발견에 쓰지 않은 환자 reference에서 확인할 수 있는가?

NIH 안의 역할 모사이며 실제 병원 비공개 코호트·독립 외부기관 일반화를 증명하는 설계로 표현하지 않는다. 사후 발견한 가설을 새 명세 이후 확인하는 연구라고 명시한다.

## 6. 평가기 검증은 어디서 할 것인가

이전 evaluator preflight448명 중 잠긴 CVPR calibration/test124명을 제외한 **324명**을 별도 evaluator-validation 후보로 둘 수 있다. 이 환자들의 현재 보유영상 전체를 metadata로 조사하면 폐기종15명·기흉45명, 정상172명·흉수99명이다. 이는 과거 preflight에서 선택한1장에 해당 label이 모두 있었다는 뜻이 아니다. 새 검증의 환자당 영상 선택은 따로 고정해야 한다.

이324명은 과거 소비된 공개자료이고 일부는 모델 학습/개발에도 사용됐다. **평가기의 동작 점검용**이며 새 생성 reference나 최종 확인으로 재사용하지 않는다. 새8,426명과는 환자가 분리돼 있다. 폐기종 양성15명은 추정 불확실성이 크므로 여기서 높은 점수가 나오면 평가기가 완성됐다고 하지 않는다.

PadChest-only DenseNet 후보는 그대로 유지한다. 다음 한 단계는 **고정 평가기 실행·전처리/출력 검산 및 조건 구별력 점검20–35분**이다. 환자 단위 양성/음성 선택, raw ranking score와 AUC/AP, threshold 미선택, 정상/흉수 대조를 실행 전에 고정한다. 이 자료만으로 판별력이 불명확하면 새 target-development의 별도 검증 부분집합을 사전에 분리할 수 있지만, reserved confirmation을 쓰거나 좋은 환자만 골라 보충하면 안 된다.

실제 생성 artifact에 대한 취약성 점검과 생성비교 명세도 필요하다. 이번에는 classifier를 로드하거나 score를 계산하지 않았다. 자료와 평가기의 관문을 통과한 뒤에만 backbone/public/private-only/pooled의 새 paired 비DP 생성을 한다. Patient-DP 진입은 실제 private 추가 효용을 확인한 뒤다.

## 7. 검산·기록

출력: `code_working/_reports/patient_usage_audit_20260917_v1`.

- 실제 감사35.649초, 독립 검산1.565초. 분석 코드와 별개로 원본 metadata·runtime units·patient sets·SHA 정렬·condition image 선택을 다시 계산했다.
- 독립 검산 **100,948항목 PASS**. 이는 파일/metadata/행별 재집계 확인 횟수이며 통계적 증거량이 아니다.
- 기존192장 생성물과 과거 public backbone 실패 adoption 파일은 변경되지 않았다. 모든 읽은 source file의 hash 불변을 확인했다.
- Audit SHA256: `5686be2b4fe49a329e5f21f5435c95d0deb144aa071bf9f3b04224c9c9a22865`.
- Contract SHA256: `8758464a9a8c37b6e8e0ac06a477fe6913a37d4d6640c260ddf73636994f18cb`.
- 최초 실행은 새 폴더 생성 시 OS 접근 거부로 멈췄고 결과물/프로세스가 없음을 확인했다. 사용자가 접근 권한을 바꾼 후 동일 코드로 실행했다. 결과를 본 뒤 pool/salt/조건을 바꾸지 않았다.
- 원격main 확인/push는 이번 감사 범위에 포함하지 않았다. 이 문서는 로컬 이력과 실행 검산에 근거한다.

**현재 결론: 별도 CVPR 평가자료를 구성할 실질적 근거가 생겼다. 자료 병목은 해소할 수 있지만, 평가기 판별력과 private 생성 효용은 아직 확인하지 않았다.**
