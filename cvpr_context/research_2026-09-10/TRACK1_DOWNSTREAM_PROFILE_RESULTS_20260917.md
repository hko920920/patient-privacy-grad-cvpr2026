# Downstream 제한 실행 결과: 생성 통과, classifier 자료 연결 오류와 제한 내 수정

2026-09-17 · 큰 단계2 / 방향1. **종합 판정은 혼합이다. 생성 연결은 통과했지만, 최초 classifier 7개 비교군 실행은 자료 혼선으로 실패했다. 원인을 고쳐 배치 구성과 실제 학습 2회를 검증했으며, 수정된 7개 비교군 전체 재실행은 아직 하지 않았다.** 방법 성능이나 사적 추가 효용의 성공 결과는 아니다.

[통합 연구 명세](TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md)와 [이번 실행 명세](TRACK1_DOWNSTREAM_PROFILE_PROTOCOL_20260917.md)를 따른 한정 profile이다. 준비 문서만 작성한 것이 아니라 실자료11,277장을 확인하고,28장 생성·3회 생성 재검사와 총100회 optimizer update를 실제 실행했다. Expert final532명과 reserved4213명의 영상·예측에는 접근하지 않았다.

## 무엇이 좋고 무엇이 나쁜가

| 질문 | 이번 결과 |
|---|---|
| 실제 실자료 파일과 개발 역할이 올바른가 | 좋음.11,277장 SHA·bytes·decode·크기·mode 확인, 네 역할 간 환자/영상/실제 SHA 중복0 |
| Private-only head가 올바르게 sampling에 연결되는가 | 좋음. 저장 입력의 feature/residual 재현 통과 |
| W=0 복원과 원상복구, 같은 입력의 private 재생성이 같은가 | 좋음. 전체 trajectory 주요 배열과 최종 출력 exact |
| 최초 classifier 7군의 자료 구성이 맞았는가 | **나쁨. 실자료 public이 합성 public으로 덮어써져 실패** |
| 오류를 수정했는가 | 수정됨. 분리 namespace,49개 CPU 배치 계획 검증, S1 한 step씩2회 exact 재현 |
| 수정된7군 전체 학습 재현을 완료했는가 | 아님.100회 제한 중98회를 이미 사용하여 수정 경로에는2회만 사용 |
| S2/S3가 S1·R1보다 좋은가 | 이번에는 AUROC/AP를 계산하지 않아 미판정 |
| 본512장·21개 classifier·새 patient-DP·expert final을 실행했는가 | 모두 미실행 |

이번 profile은 오류를 찾아 고치는 데는 도움이 됐다. 그러나 오류를 수정했다는 이유로 최초98회 실행을 유효한 비교로 바꾸지는 않는다. **Classifier 전체 연결 관문은 아직 완료가 아니다.**

## 실자료와 사용 경계

| 역할 | 환자 | 영상 |
|---|---:|---:|
| 공개 실자료 | 672 | 813 |
| 사적 역할 실자료 | 80 | 320 |
| Classifier selection | 2,027 | 5,097 |
| Method development | 2,026 | 5,047 |
| 합계 | 4,805 | 11,277 |

총4,698,760,078bytes를 읽어 실제 SHA와 PIL decode를 확인하고, 전체 영상을 종횡비 유지 grayscale224 letterbox로 캐시했다. 이 과정은42.217초였다. 네 역할의6개 조합 모두 환자 ID·image ID·검증된 파일 SHA 교집합이0이다. 별도 코드로64장의 raw 전처리를 다시 계산해 exact를 확인했고, 캐시11,277행의 tensor hash도 대조했다.

새 개발4053명의 pixel은 이제 이 profile에서 소비한 상태다. 두 개발군에서 hash로 고른128장에 짧은 profile 모델의 비용 측정용 추론도 했다. 해당 예측은 저장했지만 성능 지표·선택에는 사용하지 않았다. 이후 이를 “아직 pixel/모델에 전혀 사용하지 않은 개발자료”라고 부르지 않는다. 개발 역할은 원래 용도대로 유지하며, final과 reserved 잠금·원 졸논 역할은 변경하지 않았다.

## 생성은 실제로 검증됐다

E4 SHA `7bc5ce421cc02091e8c3812676e9c27dc58330afce5e9384abd8edc9f61a3b91`, CFG7.5, FP32, DDIM30,256px를 유지했다. 네 새 latent를 네 방법에 공유하여 기흉4cell·normal/effusion/cardiomegaly 각1cell, 총28장을 생성했다. W0·base 복구·private-only 동일 입력을 각각1회 재실행해 총31decode였다.

- UNet931회 호출(배치2, 총1862example), generator backward0회.
- Private-only 저장 입력 feature/residual offline–online 통과.
- W0·복구·private 재실행의 trajectory와 최종 출력 exact.
- 독립 DDIM840전이, conditional-only correction630회, CFG7.5배 산술 확인.
- Backbone와 LoRA weight 불변, hook 잔류·NaN/Inf 없음.

Public/pooled는 원 생산 가중치를 그대로 사용했다. Private-only만 기존 독립 통계 분석의 검산된 private weight를 사용했다. 기존 독립 재계산 public/pooled와 원 생산 weight의 약1e-15 차이를 동일 파일이라고 무시하지 않았다.

[28장 전체 grid](../../code_working/_reports/downstream_profile_20260917_v3/profile_grid.png)를 이번 작업자가 직접 확인했다. 기본 흉부 형태는 보이지만 구도 유사성, 질감 변화와 artifact가 남아 있다. 방법명을 아는 단일 AI의 연결 확인이며 가림 임상 판독이나 기흉 표현 정확성 평가가 아니다. 이 grid에서 좋은 seed나 방법을 선택하지 않았다.

## Classifier에서 발견한 실제 오류

최초 구현은 실자료 pool에 `public`이라는 키를 쓰면서, 합성 방법 이름에도 `public`을 사용했다. 합성 pool을 dictionary에 추가할 때 공개 실자료 pool이 덮어써졌다.

그 결과 R0/R1은 공개 실자료 대신 public-head 합성자료로 학습했고, S0–S3/Dreal도 공통 real16 부분에 public-head 합성자료가 들어갔다.7군×7step×2회=98update의 입력·logit·loss·최종 state는 반복 간 exact였지만, **잘못된 자료로 정확히 반복된 것**이다. 독립 검산의 `Shared public half` 검사가 이를 실패 처리했다. 기존 classifier metric을 계산하거나 방법 결과로 채택하지 않았다.

수정본은 `real/public`, `real/private`, `synthetic/backbone`, `synthetic/public`, `synthetic/private_only`, `synthetic/pooled`의6개 키를 사용한다. 기존 실행 코드는 덮어쓰지 않고 `data_v2.py`·`train_v2.py`로 구분했다.

수정 검사는 다음 범위다.

- 7군×7개 step의49개 배치를 CPU에서 구성하고 실자료/합성자료 출처, 환자·영상·label, class balance와 공통 real draw를 검산했다.
- S1을 같은 초기값에서 한 step씩2회 실행했다. 실제 공개16장+합성16장, 입력tensor·logit·loss·최종 가중치 exact를 확인했다.
- 총 업데이트는 잘못된98회+수정된2회=100회다. 수정본의7군 전체 학습을 추가 실행해 제한을 넘기지 않았다.
- 환자별 영상 수가1장 대9장인 의미 검사는 두 환자가 약0.5씩 선택되고, CSV를5회 복제해도 같은 draw가 나오는 것을 확인했다. 이것은 환자 균등 sampling 구현 확인이지 성능 증거가 아니다.

따라서 수정본에 대한 결론은 **모든 군의 자료 선택은 확인했고, 실제 수정된 학습은 S1의1step 재현까지만 확인했다**이다. 전체 비교군의 실학습 정합 확인을 완료했다고 쓰지 않는다.

## 실제 비용과 본실험 예상

| 항목 | 실측 |
|---|---:|
| 전체 raw 검증·224 cache | 42.217초 |
| Backbone 생성 | 2.600초/장 |
| Public head 생성 | 2.638초/장 |
| Private-only head 생성 | 2.620초/장 |
| Pooled head 생성 | 2.631초/장 |
| Trace·PNG 저장 | 평균0.422초/장 |
| 생성 peak allocated / reserved | 3.930 /4.322GiB |
| 31decode·witness·모델로딩·저장 포함 생성 단계 | 108.827초 |
| 수정된 S1 두 번째1step | 0.155초 |
| 수정된 S1 peak allocated / reserved | 0.837 /1.182GiB |
| 독립 저장 결과·수정 검산 | 9.327초 |

최초 잘못된7군 실행에서 warmed step 중앙값은0.102–0.137초였다. 이는 같은 ResNet18 연산의 거친 처리량 참고일 뿐, 올바른7군 비용 비교로 사용하지 않는다. 수정된 S1의0.155초도 warm 측정1개여서 장기 처리량을 확정하지 않는다. 생성 overhead의 작은 차이는7장/방법의 순차 profile이며 통계적 우열이 아니다.

512장 생성 연산은 약22.4분, 동일한 상세 trace 저장까지 유지하면 약3.6분이 추가된다. Classifier21run의400/800step, 공개 calibration1600step, 개발 forward를 단순 외삽한 기계 처리량은 약49–67분이지만 classifier 입력 오류와 짧은 측정 때문에 **확정 ETA가 아니다**. 수정본 전체 확인 후 다시 갱신하며, 본 비DP 패키지는 orchestration·분석·검산을 포함해 대략90–150분을 계획 범위로 둔다. 새 DP나 expert final 비용은 포함하지 않는다.

## 실패·수정 기록을 보존한 방식

1. 준비 시 독립 재계산본과 원 weight의 차이를 발견했다. 원 public/pooled를 유지하고 실패 source를 보존했다.
2. V1은 SHA 문자열 대소문자 비교 오류였다. 같은 bytes/digest임을 확인하고 입력 digest만 lowercase로 정규화했다. 부분 읽기의 정확한 파일 수는 알 수 없으며 새 모델 실행은0이었다.
3. V2는11,277장 파일 검증을 완료한 뒤 일반 추출 row에 conditioning이 있다는 잘못된 가정으로 생성 준비가 실패했다. 모델 추론0회였다. 완전한 과거 train witness와 공개 backbone의 고정 prompt cache를 사용하도록 V3를 동결했다. 검증된 실자료 cache는 hash를 확인해 그대로 재사용했다.
4. V3 생성은 통과했고 classifier98회에서 자료 키 충돌이 발견됐다. 원 계약·실행·실패를 유지한 채 별도 수정 계약으로2회만 추가했다. 검산 결과의 최상위 판정도 PASS로 바꾸지 않고 `MIXED_GENERATION_PASS_ORIGINAL_CLASSIFIER_SOURCE_FAIL_REPAIR_LIMITED`다.

어떤 수정도 prompt·seed·checkpoint·head scale·수치 tolerance·효용 기준을 결과에 맞춰 변경한 것은 아니다. 다만 여러 구현 오류가 있었으므로 “처음부터 오류 없이 실행됐다”고 쓰지 않는다.

## 연구 판단과 다음 한 단계

현재 확인하려는 중심 질문은 여전히 **method-development에서 S2 또는 S3가 S1 및 R1보다 실제 downstream 효용을 추가하는가**다. 이번에는 그 성능을 측정하지 않았다. 기존192장 pooled 효용 미통과, PadChest·CheXzero 실패, expert label의 공개 가공본 provenance 한계도 그대로 유지한다.

다음은 **수정된 kernel로7군×1step×2회=14update만 재검증하는 별도 한정 실행**이다. 새 생성 없이 기존 profile 입력을 사용하고, 이번100회와 구분하여 계약·소비량을 기록한다. 예상10–15분(실제 GPU 학습은 짧으며 명세·독립 검산·상태 기록 포함)이다. 통과하면 본512장과 공개 classifier calibration·21개 비DP 비교를 실행할 기반이 된다. 이 보고서는 그14회나 본실험을 이미 완료했다고 승인/기록하지 않는다.

비DP가 개발 관문을 넘을 때만 의료 backbone patient-DP와 DP-LoRA에 투자한다. Expert final은 모든 비DP·DP 모델과 분석을 동결한 후 한 번 평가한다. 강한 real augmentation, 두 번째 ImageNet classifier, 넓은 negative prompt 및 독립 합성 bank는 본 결과가 유망하면 final 전 개발 단계에서 보강한다. `final_ready=false`다.

## 산출물

- [실행 코드](../../code_working/downstream_utility/run.py), [수정된 자료 선택](../../code_working/downstream_utility/data_v2.py), [수정 학습 kernel](../../code_working/downstream_utility/train_v2.py).
- [V3 실행 계약](../../code_working/_reports/downstream_profile_20260917_v3/contract.json), [생성 결과](../../code_working/_reports/downstream_profile_20260917_v3/generation.json).
- [Classifier 최초 실패](../../code_working/_reports/downstream_profile_20260917_v3/classifier_verification_failure.json), [제한 내 수정 결과](../../code_working/_reports/downstream_profile_20260917_v3/classifier_repair_v2/result.json), [독립 검산·혼합 판정](../../code_working/_reports/downstream_profile_20260917_v3/verification_v2.json).

위 자료는 로컬 실행 기록이다. 이번 작업에서 원격 main 확인이나 push를 수행했다는 뜻은 아니다.
