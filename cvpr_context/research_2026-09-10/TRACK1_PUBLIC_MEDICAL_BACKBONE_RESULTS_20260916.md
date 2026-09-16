# 방향1: 공개 의료 LoRA 실제 학습·52장 생성 결과

**최종 판정: 현재 구성은 채택 기준에 미달했다.** 공개 역할 640명·749장으로 새 LoRA를 실제 학습했고 흉부 X-ray 형태는 만들었다. 그러나 사전 선택에서 E4/E8 각각 13/16장이 통과한 뒤, 선택된 E4의 별도 확인 입력에서는 합의 통과가 **10/16장**, 흉수 prompt의 네 입력에서는 **0/4장**이었다. 전체 12/16 이상 및 모든 prompt에서 2/4 이상이라는 고정 기준을 충족하지 못했다.

따라서 현재 E4를 다음 작은 head 비교의 기반모델로 승인하지 않는다. E8로 확인 모델을 바꾸거나, 확인 seed를 다시 고르거나, 추가 학습·DP solver를 자동 실행하지 않았다. 현재 큰 단계2·방향1을 유지한다. **공개자료만으로 기본 흉부 형태를 만들 수 있다는 근거는 확보했지만, 안정적인 생성 품질이나 사적 추가 효용은 확보하지 못했다.**

## 1. 이번에는 실제로 무엇을 실행했나

[동결 계획](TRACK1_PUBLIC_MEDICAL_BACKBONE_PLAN_20260916.md)의 한 패키지를 실행했다. 계획의 자료 역할, 훈련량, 입력, 통과 기준을 결과 이후에 변경하지 않았다.

| 항목 | 실제 실행 |
|---|---|
| 공개 역할 학습 자료 | NIH PA, 640명·749장; 기존 보호·평가·보조 역할과 환자 중복 제외 |
| 초기화 | 고정 revision의 일반 SD2.1에서 새 rank8/alpha8 LoRA; M1/M2 미사용 |
| 캐시 | 학습 749장만 새 VAE latent mode로 인코딩; 새 공개 prompt embedding |
| 학습 | 한 번의 실행, batch4, AdamW 1e-4, 성공 업데이트 1,498회 |
| 시도/실패 | 1,498회 시도, 실패·overflow retry 0회 |
| E4 | 749회 업데이트, 모든 영상 정확히 4회, 총 2,996회 반영 |
| E8 | 1,498회 업데이트, 모든 영상 정확히 8회, 총 5,992회 반영 |
| 가중치 | frozen base FP16, 학습 LoRA FP32 1,659,904개 파라미터 |
| 생성 | FP32, autocast 없음, CFG1, DDIM30, eta0, 256×256 |
| 실제 생성량 | 선택32장 + 이전 입력 진단4장 + 새 확인16장 = **52장** |
| 새 사적 head/환자-DP | 실행하지 않음 |

749장은 4로 나누어떨어지지 않으므로, 고정된 epoch 순열을 연결한 뒤 전체를 batch4로 나눴다. 마지막 영상을 버리거나 epoch마다 중복해서 채우지 않았다. E4/E8의 영상별 노출 횟수를 실제 로그와 checkpoint에서 확인했다. 공개 학습은 image-uniform이며 환자-DP 학습이라고 부르지 않는다.

## 2. 선택과 별도 확인 결과

그림의 모델 이름과 prompt/seed 정보를 숨기고 두 assistant가 비임상 형태 판독을 했다. 둘 다 통과한 경우만 합의 통과이며, 모호함 또는 불일치는 미통과다. 평가 대상은 정면 흉부 형태와 큰 비의료 패턴·회전·중첩·구조 왜곡 여부다. 질환 정확성은 판독하지 않았다.

| 단계·모델 | Generic | Normal | Effusion | Cardiomegaly | 전체 | 결정 |
|---|---:|---:|---:|---:|---:|---|
| 선택 E4 | 4/4 | 4/4 | 3/4 | 2/4 | **13/16** | 통과·먼저 도달한 E4 선택 |
| 선택 E8 | 4/4 | 4/4 | 3/4 | 2/4 | **13/16** | 통과; 별도 확인 대상으로 교체하지 않음 |
| 새 입력 확인 E4 | 3/4 | 4/4 | **0/4** | 3/4 | **10/16** | **미통과** |

확인 단계에서 두 판독자 각각의 통과 수는 11/16장이었다. 합의 통과는 10장, 판독 불일치는 2장이었다. 따라서 최종 미통과는 단지 합의 규칙 때문에 생긴 결과가 아니다. 각 판독자의 전체 통과 수도 12장에 못 미쳤다. 선택 단계 불일치는 32장 중 2장이었다.

**판독 가림의 한계:** 선택 단계에서는 두 판독자가 checkpoint 매핑을 열기 전에 표를 기록했다. 확인 단계의 독립 판독자는 선택 checkpoint와 매핑을 열지 않았다. Root는 E4를 선택하고 실행한 조정자이므로 그 모델 정체를 이미 알았다. 확인 그림의 label과 task 매핑은 숨겼지만, 확인 전체를 ‘두 명 모두 모델 정체를 모르는 완전한 blind 판독’이라고 주장하지 않는다. 이 한계는 root 원표에도 명시했다.

### 선택 E4: 16장 전체

![선택 E4 전체와 합의 판정](../../code_working/_reports/public_medical_backbone_20260916_v1/selection_E4_reviewed.png)

### 선택 E8: 16장 전체

![선택 E8 전체와 합의 판정](../../code_working/_reports/public_medical_backbone_20260916_v1/selection_E8_reviewed.png)

### 별도 확인 E4: 16장 전체

![별도 확인 E4 전체와 합의 판정](../../code_working/_reports/public_medical_backbone_20260916_v1/confirmation_E4_reviewed.png)

### 이전 입력 진단 4장: 통과율 집계에서 제외

![과거 prompt·초기 latent를 재사용한 진단 4장](../../code_working/_reports/public_medical_backbone_20260916_v1/diagnostic_four.png)

이 네 장은 과거 prompt와 실제 initial latent를 재사용했다. conditioning은 이번 공개 캐시에서 다시 만들었으므로 과거 embedding까지 비트 단위로 같은 입력이라고 하지 않는다. E4/E8 사이에서는 실제 latent와 conditioning이 같다. Root가 네 장 모두 확인했으며 흉부 형태는 있지만, 특히 effusion 그림에는 큰 중첩·왜곡이 남았다. 이 판독은 별도 정량 통과율로 합치지 않았다.

## 3. 실패가 무엇을 뜻하나

앞선 일반 SD2.1+작은 출력 head의 비흉부 출력과 달리 이번 새 공개 LoRA는 많은 입력에서 흉부 구조를 만들었다. 사적 역할 M1을 공개 기반모델로 재사용하지 않고도 도메인 형태를 만들 수 있다는 점은 긍정적이다.

그러나 확인 영상에는 큰 검은 사각·띠에 의한 구조 단절, 전면적인 음각/지도 같은 질감, 반복 도형과 하부 중첩, 모호한 구조 왜곡이 관측됐다. **이번 구성은 형태를 만들 수 있으나, 새 입력에서도 충분히 안정적이라는 진행 조건을 충족하지 못했다.** 이 상태에서 작은 사적 head의 미세한 효용이나 DP 손실을 비교하면 기반모델의 큰 생성 불안정성이 주된 결과를 가릴 수 있다.

흉수 prompt의 확인 네 입력이 모두 미통과였다는 사실은 기록하되, 이것만으로 흉수 prompt 자체가 원인이라고 확정하지 않는다. prompt별 seed가 다르므로 prompt 효과와 seed 효과를 분리한 대조가 아니다. 약한 라벨, 공개 자료의 양/분포, attention LoRA의 표현 범위, CFG1과 sampling 조건 가운데 무엇이 주원인인지는 이번 실행으로 분해하지 않았다.

이 결과는 다음 주장도 뒷받침하지 않는다.

- 환자-DP 연구 또는 공개 의료 기반모델 활용 자체가 불가능하다.
- 학습량만 늘리면 문제가 해결된다.
- E8은 모든 새 입력에서 E4와 같거나 더 나쁘다. E8의 확인 입력은 실행하지 않았다.
- 사적 자료의 추가 생성 효용이 없다. 이번에는 사적 적응 비교에 진입하지 않았다.
- 지금 작은 head나 DP solver를 바꾸면 큰 생성 artifact가 해결된다.

## 4. 정확성과 provenance 확인

| 검산 단계 | 결과 | 범위 |
|---|---|---|
| 학습 저장 기록 | PASS, 24,684개 확인 | 원 계획/역할 경계, train-only cache, 초기 zero-B, 첫 입력, 1,498회 시도, checkpoint/optimizer/scaler, 노출 |
| 선택 생성 | PASS, 20,139개 확인 | 36장, 실제 공통 입력, 1,080개 DDIM 전이, CFG1/후처리/PNG, 원본 hash |
| 확인 생성 | PASS, 12,991개 확인 | 선택 결속, 새 16개 입력, 480개 DDIM 전이, 후처리/PNG, 원본 hash |

초기 zero-B adapter의 활성/비활성 예측은 정확히 같았다. 학습 전후 동결 base hash가 같고, 생성에 실제 로드된 adapter는 해당 FP32 checkpoint와 정확히 같으며 생성 후에도 변하지 않았다. 두 checkpoint가 같은 prompt별 실제 initial tensor와 conditioning을 사용했는지도 확인했다.

검산은 별도 CPU 구현으로 저장 입력·상태·DDIM 산술·PNG·provenance를 확인한 것이다. UNet/VAE 전체를 두 번째 독립 모델로 다시 추론하거나 1,498회의 gradient를 재학습한 것은 아니다. 확인 항목 수는 통계적 표본 수나 성능 증거가 아니다. 최종 형태 판정은 위의 제한된 비임상 시각 검토다.

## 5. 실제 시간과 연산

| 항목 | 실측 |
|---|---:|
| 새 공개 캐시 인코딩·저장 | 28.267초 |
| 전체 1,498회 학습 loop·저장 | **696.330초, 약 11분 36초** |
| E4 저장 시점 | 학습 시작 후 약 349.5초 |
| 선택·진단 36장 생성 단계 | 109.610초 |
| 확인 16장 생성 단계 | 50.627초 |
| 학습 최대 GPU allocated memory | 2,073,024,000 bytes, 약 1.93GiB |
| GPU | RTX 3070 8GB |
| 학습 UNet 호출/역전파 | 1,498 / 1,498 |
| 초기 동일성 확인 UNet 호출 | 2 |
| 생성 UNet 호출 | 1,560 |
| 전체 새 UNet 호출 | **3,060** |
| VAE 인코딩 / 생성 decode | 749개 기록(188 batches) / 52회 |

UNet 호출 수는 API forward 호출 수다. gradient checkpointing으로 역전파 중 재계산하는 내부 연산까지 별도 forward로 세지는 않았다. 학습 비용과 메모리는 새로운 공개 LoRA 학습의 값이며 사적 head 효율 우위로 사용할 수 없다. 각 단계 시간은 코드에서 측정한 범위이며 프로세스 시작, 일부 모델 로딩·hash 검사, 구현·판독·문서 시간을 모두 포함한 end-to-end 측정치가 아니다.

학습 후 E4를 골랐지만 실제 실험에서는 E8까지 한 번 학습했다. 이번 실행 비용을 E4의 349.5초만으로 보고하지 않는다. 시작 전 전체 작업 예상은 50–85분, 순수 학습 예상은 10–16분이었다.

## 6. 자료와 결론의 범위

- NIH 공개자료 안의 역할 분할이다. 실제 외부 병원의 비공개 코호트로 일반화했다고 주장하지 않는다.
- 공개 선택/확인 환자 128명씩은 후속 실제 참조 평가용으로 보존됐다. 이번 16개 확인 단위는 새 생성 prompt/seed 조합이며 그 128명에 대한 환자 평가가 아니다.
- 두 판독자는 assistant이며 임상 전문가가 아니다. threshold는 사전에 고정한 자원 배분용 경험적 기준이다.
- 영상의 임상 타당성, 조건별 질환 반영, diversity, memorization, 정량 생성 품질을 이 관문에서 인증하지 않았다.
- 환자-DP, public-only 대비 private added utility, 강한 DP 방법 대비 품질–비용 우위는 여전히 미검증이다.

## 7. 다음 한 단계

**현재 공개 E4를 기반으로 새 작은 head/DP 비교를 진행하는 결정은 보류한다.** 이번 결과로 공개 도메인 학습이 기본 형태를 만든다는 것은 확인했으므로, 다음은 필요한 생성 품질에 맞는 공개 기반모델과 운용 조건을 정하는 좁은 설계 검토다.

이미 의료 생성 성능을 보여 준 공개 모델의 자료 경계·사용 조건·로컬 호환성을 확인하고, 현재 새 공개 LoRA의 명확한 실패 형태와 sampling/conditioning 설정을 대조한다. 현재 결과를 근거로 선택 가능한 출발점과 필요한 최소 대조를 먼저 고정한다. 예상 **20–40분의 자료·코드 검토**이며 새 GPU 학습량/실험은 그 설계에서 별도로 산정한다. 이 보고서에서 해당 후속 조사를 완료했다고 주장하지 않는다.

## 8. 재현 파일

- [실행 contract](../../code_working/_reports/public_medical_backbone_20260916_v1/contract.json) — SHA256 `ca3c0b07efa070f9eb6efe5a94de49cc6d7a9fd39ed551c51ad98b130a0ef681`
- [실행 코드](../../code_working/public_medical_backbone/run_pilot.py), [독립 수치 검산기](../../code_working/public_medical_backbone/verify_pilot.py), [판정 집계기](../../code_working/public_medical_backbone/review_gate.py)
- [학습 결과](../../code_working/_reports/public_medical_backbone_20260916_v1/training_report.json), [학습 검산](../../code_working/_reports/public_medical_backbone_20260916_v1/training_verification.json)
- [선택 manifest](../../code_working/_reports/public_medical_backbone_20260916_v1/manifest_selection.json), [선택 판정](../../code_working/_reports/public_medical_backbone_20260916_v1/selection_decision.json), [선택 검산](../../code_working/_reports/public_medical_backbone_20260916_v1/selection_verification.json)
- [확인 manifest](../../code_working/_reports/public_medical_backbone_20260916_v1/manifest_confirmation.json), [확인 판정](../../code_working/_reports/public_medical_backbone_20260916_v1/confirmation_decision.json), [확인 검산](../../code_working/_reports/public_medical_backbone_20260916_v1/confirmation_verification.json)
- [선택 root 원표](../../code_working/_reports/public_medical_backbone_20260916_v1/review_selection_root.json), [선택 독립 원표](../../code_working/_reports/public_medical_backbone_20260916_v1/review_selection_independent.json)
- [확인 root 원표](../../code_working/_reports/public_medical_backbone_20260916_v1/review_confirmation_root.json), [확인 독립 원표](../../code_working/_reports/public_medical_backbone_20260916_v1/review_confirmation_independent.json)

모든 checkpoint·원 PNG·trajectory·시도 로그를 로컬 실행 폴더에 보존했다. 이전 실험 결과·코드·동결 계획은 수정하지 않았다. 원격 커밋 갱신을 주장하는 보고서가 아니다.
