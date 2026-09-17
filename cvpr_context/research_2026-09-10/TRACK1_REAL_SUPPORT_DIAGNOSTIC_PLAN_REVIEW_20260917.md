# 실자료 학습 범위 대조: 명세와 예정 노출 점검

2026-09-17. **대조를 설계할 자료와 실행 기반은 있다. 성능은 아직 미실행이다.** 기존 생성기나 분류기를 바꾸기 전에, 같은 학습 절차가 더 넓은 실환자 pool에서 일반화할 수 있는지를 묻는 대조 하나로 고정했다.

본 작업은 [실행 명세](TRACK1_REAL_SUPPORT_DIAGNOSTIC_PROTOCOL_20260917.md)·[계획 JSON](spec_sources/real_support_diagnostic_plan_20260917.json)·예정 명부/배치/역할 overlay를 작성하고 metadata로 검증한 것이다. **새 학습·추론·생성·raw pixel decode는 모두0**이다. 기존 head와 LoRA의 효용 실패는 그대로다.

## 1. R1 대 Rwide 하나

| 항목 | 기존 R1 | 예정 Rwide |
|---|---:|---:|
| 학습 가능 환자 | 672 | 2,699 |
| 학습 가능 영상 | 813 | 5,910 |
| 기흉 양성 환자 | 6 | 120 |
| 기흉 양성 영상 | 6 | 236 |
| class-balanced batch | 32, 양/음성16씩 | 동일 |
| step·classifier seed | 400·11/23/37 | 동일 |
| 자료 유형 | 공개 실영상 | 기존 공개+former-selection 실영상 |
| 모델·전처리·학습 설정 | 기존 ResNet18 recipe | 동일 |

추가 집합은 기존 classifier-selection2,027명·5,097장 전체다. 영상 외관, 개별 loss/score, 질환별 성능에 따라 환자를 고르지 않았다. 이 집합에 양성 환자114명·영상230장이 있어, 기존 public6명과 합쳐120명이다.

원 졸논 역할은 추가 환자 전부 private_train이다. 따라서 **Rwide는 비DP 실자료 진단이며 확대된 공개 baseline이 아니다.** 기존 private80을 추가하는 Dreal과도 다르다. Public813, head public32/64, private80/320이라는 구분은 유지한다.

## 2. 무엇을 고정하고 무엇을 바꾸는가

R1에서 사용한 ImageNet 초기화·세 seed·FP32·224 letterbox·affine 증강·AdamW lr1e-4·weight decay1e-4·400step을 고정한다. 새 학습률/epoch 선택이나 BN 재보정은 하지 않는다.

바뀌는 것은 **두 half-batch가 추출하는 자료 pool**이다. Rwide는32개 slot 모두 합친 pool에서 기존 규칙대로 추출한다. 앞16장을 기존 public로 고정하는 방식이 아니며, 공개/추가자료를50:50으로 섞는 방식도 아니다.

Class별 환자 균등 추출 후 해당 환자의 해당 class 영상을 균등하게 뽑는다. 양성/음성 보유 환자120/2684명, 두 label 영상을 모두 가진 환자105명이라는 구조를 그대로 둔다. 환자 any-positive로 음성 방문을 다시 라벨링하지 않는다.

따라서 동일한 총 step/slot 아래 환자 범위, 고유 양성 support, 영상 구성, 반복 노출 및 source 질량이 함께 달라진다. 양성 수 하나만의 순수 인과 효과를 분리하는 설계는 아니다.

## 3. 실제 실행 전 예정 배치 계산

기존 RNG salt·정렬·PCG64·slot 규칙을 유지해3seed×400step×32 = **38,400개 예정 slot**을 만들었다. 아직 optimizer에 들어간 slot은0개다.

| classifier seed | 예정 고유 환자 | 예정 고유영상 | 기존 공개 양성6장의 영상별 노출 | 추가 양성114명의 환자별 노출 |
|---|---:|---:|---:|---:|
| 11 | 2,466 | 3,439 | 47–62 | 37–74 |
| 23 | 2,468 | 3,459 | 45–61 | 30–80 |
| 37 | 2,472 | 3,476 | 48–58 | 36–74 |

세 seed 모두 양성120명·양성236장이 예정 배치에 전부 등장한다. Eligible5910장 전체를400step에 빠짐없이 학습하는 것은 아니다. 실제 학습 후 trace가 이 일정과 일치하는지 다시 확인해야 한다.

기존 R1에서는 같은 공개 양성6장이 영상당 대략1,025–1,113회 사용됐다. Rwide에서는45–62회다. 추가 집합의 실제 예정 slot 비중은 seed별 약85.20%/84.80%/85.54%이며, 전체 환자 수 비중이나50%와 같지 않다. class-balanced sampling의 결과다.

이 반복 감소가 성능을 개선할지는 미확인이다. 더 넓은 pool에서 같은400step만 쓰므로 학습의 충분성도 보장하지 않는다.

## 4. 역할 전환의 경계

이2,027명은 이미 기존 R1의 네 LR/step 후보를 평가한 집합이다. 미사용 validation이나 새 환자 확인집합으로 부르지 않는다. 이번에는 새 학습을 아직 하지 않았으며 별도 overlay의 상태는 **planned_not_yet_optimizer_consumed**다.

향후 Rwide 실행 때 그 pool을 해당 분기의 학습 자원으로 전환하고 실제 등장 환자·영상을 누적 기록한다. 그 뒤 같은 환자를 Rwide의 validation 또는 독립 confirmation으로 되돌려 쓰지 않는다. 원본 졸논/CVPR 역할 CSV와 과거 calibration 결과는 바꾸지 않는다.

| 비교 대상 | 확대 pool과 환자/영상/기록 file SHA 교집합 |
|---|---|
| Method-development2,026명·5,047장 | 0 / 0 / 0 |
| 기존 private80·320장 | 0 / 0 / 0 |
| Expert final532명 | 환자0, 영상0; 기존 metadata 감사 유지 |
| Reserved4,213명·잠긴 졸논 역할 | 환자0 |

원본 파일의 SHA/decode를 이번에 다시 수행한 것은 아니다. 명부의 기록 hash와 과거 원본 검증을 사용했다. 실행 전에는 필요한 실제 파일과 cache 입력을 다시 결속해야 한다.

개발 환자 집합은 환자 단위로 분리돼 있지만, 이미 기존 모델 결과를 여러 번 본 자료다. 이번 계획이 개발 결과를 본 뒤 제안됐다는 이력과 calibration 집합의 새 학습 사용 이력을 모두 남긴다. Expert final·reserved는 열지 않는다.

## 5. 판정과 해석

Primary는 Rwide−R1의 seed별 AUROC 차이를 평균한 값이다. AP, seed 방향, 개발 class별 BCE/logit 및 실제 본 학습영상 성능도 보고한다. R0 대비는 사전 secondary다. 기존 patient-cluster2,000 draw를 공유해95% 구간을 계산하되 개발 기술 통계로 제한한다.

R1 대비 평균AUROC +0.01 이상,2/3seed 양의 차이, 평균AP 비감소를 계속 검토할 개선 후보 표시로 고정했다. 이 값은 임상적 기준이나 통계적 유의성, 충분한 절대 성능의 기준이 아니다. **통과해도 DP·합성자료 효용 관문 통과가 아니다.**

- 반복적인 개선: 같은 학습 절차에서 이 자료 범위 변화가 도움이 되는 개발 근거. 어떤 요소가 원인인지는 추가 분리가 필요하다.
- 작은 변화·seed 불일치·AP 악화: 혼합/불확실. 자동으로 다른 subset·epoch·seed를 찾지 않는다.
- 미통과: 확대 pool과 고정400step 조합에서도 개선 근거 미확보. 자료 확대 일반의 무용성이나 생성기/분류기 하나의 원인을 확정하지 않는다.

기존 full64·LoRA 구성의 실패를 다시 성공으로 고치거나, 원래 공개 양성6명이라는 연구 조건을 Rwide로 몰래 대체하지 않는다.

## 6. 구현 공백과 다음 한 패키지

CPU 계획 코드는 [prepare](../../code_working/real_support_plan/prepare.py), 별도 metadata 검산은 [verify](../../code_working/real_support_plan/verify.py)다. **새 runtime 공급자와 GPU 연결 검사는 아직 미구현이다.**

다음은 공급자 구현과 제한된 연결 검사다. 연결 검사는 seed11로 고정하며 기존 R1 한step 대 새 공급자의 public-only 한step(2update), Rwide 한step×2 replay(2update)를 별도 수행한다. 이4update는 본학습과 분리한다. 통과 후 새 초기화에서 Rwide3seed×400 =1,200update를 수행하고, 세 모델을 고정한 뒤 개발 결과를 계산하는 명세다.

기존 R1을 대조로 재사용하려면 kernel/source/initialization 대응과 저장 개발64장×3seed의 평가 경로 대응도 확인해야 한다. 새 runner는 기존 legacy data/train import 차단을 유지하고 실제 import 위치와 source SHA를 계약에 저장해야 한다. 연결 실패 시 과거 결과를 동일 조건이라고 가정해 진행하지 않는다.

기존 R1의 세 학습은 순수 학습시간 합계161.78초였다. 새 pool/cache/검산 시간은 별도다. 향후 구현·실행·보고의 전체 작업은30–50분 예상이며 아직 Rwide 실측시간이나 성능은 없다.

## 7. 검증과 현재 결정

[독립 검산](spec_sources/real_support_diagnostic_plan_verification_20260917.json)은124,175개의 metadata/일정 확인을 통과했다. 원 R1의1,200개 batch가 그대로 재현됐고 새1,200개 예정 batch도 별도 sampler와 일치했다. 검사 수는 성능 표본 수가 아니다.

**실행 가능한 진단 명세 하나가 준비됐다. 자료 확대의 효과나 성공 가능성을 확인한 것은 아니다.** 현재 실제 마지막 모델 결과는 고정 분류기의 일반화 진단, 마지막 방법 효용 결과는 LoRA 미통과다. 이번에는 원본 영상·모델을 새로 실행하지 않았고, DP 중단·expert/reserved 보존·큰 단계2·final_ready=false를 유지한다.
