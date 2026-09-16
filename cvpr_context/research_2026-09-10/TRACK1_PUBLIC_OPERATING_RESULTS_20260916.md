# 공개 의료 LoRA: 공통 latent·CFG 운용 진단 결과

**판정: 새 입력의 기본 형태 생성은 긍정적이다. 그러나 CFG를 높여 이전 실패를 해결했다는 증거는 얻지 못했다.** 새 네 latent를 모든 prompt·checkpoint·CFG에 공통 적용해 64장을 실제 생성했다. Root 한 명의 가림 형태 판독에서 E4/CFG1, E4/CFG7.5, E8/CFG1은 각각16/16장, E8/CFG7.5는15/16장이 개발 기준을 통과했다.

이번 입력에서는 CFG1의 실패부터 재현되지 않았다. 따라서 7.5의 성공을 CFG1 실패의 해결로 해석할 수 없다. 앞선 두 판독자의 확인10/16·effusion0/4 실패는 그대로 보존한다. **현재 모델의 채택 승인은 여전히 false이며 작은 사적 head·DP 학습은 실행하지 않았다.**

## 1. 사용자의 검토를 어떻게 반영했나

[실행 전에 고정한 명세](TRACK1_PUBLIC_OPERATING_PROTOCOL_20260916.md)를 따랐다. 일반 SD2.1+작은 head의 실패, 사적 LoRA 경로 진단, 공개 LoRA의 부분적 형태 형성과 확인 미통과를 분리했다. 공개 backbone 조정을 무한 반복하지 않도록 이번 운용 진단 한 번과 필요한 경우 강한 공개 대안 한 후보로 투자 한도를 정했다.

운영 기준 미달과 모델의 본질적 불안정성에 대한 과학적 확증도 구별했다. prompt별로 서로 다른 seed를 배정하던 문제를 고쳤고, 같은 실제 initial tensor를 네 prompt 모두에 사용했다. 다만 어떤 요인의 변경이 결과를 바꾸더라도 그 요인이 유일한 실패 원인이라고 확정하지 않기로 했다.

## 2. 실제 실행과 결과

E4/E8 × CFG1/7.5 × 새 공통 latent4개 × prompt4개 = **64장**을 생성했다. 새 학습·역전파·사적 head·DP 실행은 모두0이다. FP32, DDIM30 eta0, 256px, 같은 공개 conditional cache를 유지했다. CFG7.5용 빈 문자열 embedding 한 개만 고정 text encoder로 새 계산했다.

| 설정 | Generic | Normal | Effusion | Cardiomegaly | 전체 형태 통과 |
|---|---:|---:|---:|---:|---:|
| E4 / CFG1 | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| E4 / CFG7.5 | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| E8 / CFG1 | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| E8 / CFG7.5 | 4/4 | 3/4 | 4/4 | 4/4 | **15/16** |

동일 checkpoint·prompt·initial latent 안에서 CFG만 비교하면 다음과 같다.

| 대응 비교 | CFG1 실패→7.5 통과 | CFG1 통과→7.5 실패 | 동일 판정 |
|---|---:|---:|---:|
| E4 | 0 | 0 | 16 |
| E8 | 0 | 1 | 15 |

미통과 한 장은 E8/CFG7.5의 normal/S1이며, 하부 흉부·횡격막 주변에 카드처럼 생긴 사각 물체가 겹쳤다. 모호한 큰 비의료적 겹침은 실패로 처리한다는 고정 기준을 적용했다. 별도의 질환 진단은 하지 않았다.

### E4 / CFG1 전체

![E4 CFG1의 네 prompt·네 공통 latent](../../code_working/_reports/public_operating_diagnostic_20260916_v1/E4_CFG1.0_reviewed.png)

### E4 / CFG7.5 전체

![E4 CFG7.5의 같은 입력](../../code_working/_reports/public_operating_diagnostic_20260916_v1/E4_CFG7.5_reviewed.png)

### E8 / CFG1 전체

![E8 CFG1의 같은 입력](../../code_working/_reports/public_operating_diagnostic_20260916_v1/E8_CFG1.0_reviewed.png)

### E8 / CFG7.5 전체

![E8 CFG7.5의 같은 입력](../../code_working/_reports/public_operating_diagnostic_20260916_v1/E8_CFG7.5_reviewed.png)

## 3. 무엇이 좋아졌고 무엇은 확인되지 않았나

**긍정적 사실:** 새 공통 latent 네 개에서는 모든 prompt의 기본 흉부 형태가 대체로 유지됐다. 특히 effusion prompt가 새 입력에서도 항상 실패하는 것은 아니었다. E4에서도 이 형태를 낼 수 있어, 추가 학습이나 더 높은 CFG가 이 네 latent의 gross 형태 형성에 필수였다는 증거는 없다.

**해결하지 못한 질문:** 실패 대조군이 이번 네 latent에서 재현되지 않아 앞선 실패 원인을 prompt·guidance·latent 가운데 하나로 좁히지 못했다. 특정 latent가 모든 모델에서 ‘나쁘다’는 사실도 보지 못했다. 결과가 앞선 확인과 다른 데에는 입력 묶음과 판독자 조건의 차이가 함께 있으므로, 개선량이나 일반화 성능으로 직접 비교하지 않는다.

**질감에 관한 관찰:** CFG7.5 그림은 더 매끈하고 대비가 강하며 렌더링처럼 보이는 경향이 있었다. 동일 seed에서 prompt를 바꾼 그림들도 시각적으로 비슷했다. 이것은 비임상 시각 관찰이지, 임상 사실성 저하·mode collapse·질환 conditioning 무효를 정량 입증한 결과는 아니다. 이번에는 KID, PRDC, 임상 라벨 정확도, memorization을 계산하지 않았다.

따라서 다음처럼 쓰지 않는다.

- “CFG1 때문에 실패했고 CFG7.5로 고쳤다.”
- “새64장 대부분 통과했으니 이전 실패가 취소됐다.”
- “E8을 더 학습했으므로 좋은 backbone이다.”
- “기본 형태 통과는 실제 의료 생성 효용 또는 patient-DP 기여를 뜻한다.”

## 4. 판독과 통계적 범위

이번 개발 판독자는 root 한 명이다. 모든 모델/CFG/prompt/seed label을 가린 네 grid의64장을 확인하고, 경계 사례4장을 개별 원본으로 확인한 뒤 원표를 저장했다. 그 다음 조건 매핑을 열고 집계했다. 가린 label과 별개로 처리 조건이 그림의 질감에서 추측될 수 있다는 한계도 원표에 적었다.

앞선 pilot은 두 판독자의 합의였으므로, 두 실험의 통과 수를 같은 평가자 조건의 점수처럼 비교하지 않는다. 이번 단일 판독은 개발용 후보 지명에만 사용하며 채택 판정으로 바꾸지 않는다.

네 실제 latent를64장 전체가 공유한다. 각 설정의16개 cell은 **네 seed block**으로 묶이며 16개 또는64개의 독립 표본이 아니다. 표본4개에 해당하는 좁은 대응 진단으로 해석한다. 이 결과에 모집단 유의성이나 임상 품질 인증을 붙이지 않는다.

## 5. 수치 검산과 정정 이력

최종 저장 tensor 검산은 **22,568개 확인 PASS**다. 실제 공통 latent, 조건 embedding과 branch 순서, CFG 산술, 1,920개 DDIM 전이, PNG 후처리, model/adapter 불변과 checkpoint/cache/source hash를 확인했다. 전체 UNet/VAE/CLIP을 독립적으로 다시 추론한 검사는 아니다.

같은 initial latent의 첫 step에서 CFG1 batch1과 CFG7.5 batch2의 conditional prediction 차이는 최대 **1.073×10^-6**이었다. 사전 허용오차 `atol=rtol=1e-4` 안에 있었으며, 이후 trajectory 차이를 모델·입력이 바뀐 오류로 혼동하지 않도록 확인한 값이다.

최초 검산은 빈 문자열의 padding token 재구성에서 실패했다. 모델 폴더의 `tokenizer_config.json`은 padding을 end-of-text로 적고 있지만, 별도로 존재하는 `special_tokens_map.json`은 `!`(ID0)를 지정한다. 실제 tokenizer와 저장 토큰은 BOS49406, EOS49407, 이후0이었다. 두 설정 파일 모두 실행 전 source 계약에 포함돼 있었다.

원 검산기는 앞의 파일만 읽었다. 원본 코드와 [최초 실패 기록](../../code_working/_reports/public_operating_diagnostic_20260916_v1/verification_attempt1_failed.json)을 보존하고, 별도v2에서 special-token 설정의 우선순위만 반영했다. 생성 영상·입력·모델·수치 알고리즘·허용오차를 바꾸거나 GPU를 다시 실행하지 않았다. v2와 정정 근거 hash는 최종 검산 결과에 남겼다.

## 6. 실제 비용

| 항목 | 실측 |
|---|---:|
| 64장 실행 단계 | **197.279초, 약3분17초** |
| CFG1 한 장의 평균 sampling/decode | 2.225초 |
| CFG7.5 한 장의 평균 sampling/decode | 2.586초 |
| UNet API forward 호출 | 1,920 |
| UNet batch examples | 2,880 |
| 역전파/학습 | 0 |
| VAE decode | 64 |
| 최대 GPU allocated memory | 4,218,274,304 bytes, 약3.93GiB |
| 최종 CPU 저장 산술 검산 | 7.071초 |

CFG7.5의 API 호출당 branch는 두 개다. 처리 example 수와 wall time을 혼동하지 않는다. 이번 RTX3070 측정에서 한 장 시간이 정확히 두 배가 된 것은 아니다. 전체197.279초에는 이 실행 함수의 모델 준비·생성·저장을 포함하지만 코드 작성·판독·문서·일부 프로세스 준비는 포함하지 않는다. 시작 전 총작업 예상35–60분, GPU 생성 예상3–6분을 보고했다.

## 7. 다음 결정과 탐색 한도

사전에 정한 개발 지표를 기계적으로 적용하면 **E4/CFG7.5가 별도 확인 후보**다. 두 변경 후보가 모두 전체12/16·prompt별2/4를 충족하면 먼저 학습된 E4를 지명한다는 규칙에 따른다. 이것은 CFG 우월성의 입증이나 최종 채택이 아니다. CFG1 대조의 새 입력 통과도 이전 미통과를 취소하지 않는다.

남은 것은 이 후보에 대해 **미리 따로 고정한 새 latent4개×prompt4개,16장을 한 번 확인하고 독립적인 형태 판독을 거쳐 채택 여부를 결정하는 것**이다. 이번64장 결과를 보고 확인 seed를 새로 고르지 않는다. 확인16장은 아직 생성하지 않았다. 이 패키지는 생성·검산·독립 판독을 포함해15–25분 예상이며 실행 전에 판독 절차와 최종 모델 경계를 다시 결속한다.

확인에서도 미달하면 이 recipe의 추가 step/seed/CFG 탐색을 종료하고, 자료·사용 조건·호환성을 확인한 강한 공개 의료 기반모델 **한 후보**만 검토·시도한다. 그 대안도 확인 기준을 넘지 못하면 현재 생성 기반 방향1을 확대하지 않는다. 다른 연구 방향은 별도 논리와 근거로 결정한다.

확인을 통과하더라도 임상 효용과 작은 private head의 추가 생성 가치는 여전히 다음 질문이다. 조건부 head를 쓰면 표준 CFG에 `s*Delta_head`가 더해지므로 보정 오차와 DP noise의 출력 영향도 같이 커진다. 새 backbone·고정 운용 조건에서 특징/통계/W를 다시 만들고, 동일 CFG의 강한 비교군으로 평가해야 한다.

## 8. 재현 파일

- [고정 실행 contract](../../code_working/_reports/public_operating_diagnostic_20260916_v1/contract.json) — `756f3d2ea401181b55664aecf477c97aad690f9be7297aea886180c1d2a929c4`
- [실행 코드](../../code_working/public_medical_backbone/operating_diagnostic.py), [원 검산기](../../code_working/public_medical_backbone/verify_operating.py), [별도 정정 검산기v2](../../code_working/public_medical_backbone/verify_operating_v2.py)
- [실제 실행 결과](../../code_working/_reports/public_operating_diagnostic_20260916_v1/execution.json), [64장 manifest](../../code_working/_reports/public_operating_diagnostic_20260916_v1/manifest.json), [수치 검산](../../code_working/_reports/public_operating_diagnostic_20260916_v1/verification.json)
- [가림 판독 원표](../../code_working/_reports/public_operating_diagnostic_20260916_v1/review_root.json), [집계·후보 지명](../../code_working/_reports/public_operating_diagnostic_20260916_v1/development_summary.json)
- [이전 공개 LoRA 확인 미통과 결과](TRACK1_PUBLIC_MEDICAL_BACKBONE_RESULTS_20260916.md), [그대로 보존한 이전 채택 보류](../../code_working/_reports/public_medical_backbone_20260916_v1/adoption_status.json)

기존 실패·원표·checkpoint를 변경하지 않았다. 이번 결과는 로컬 실행과 기록이며, 사용자가 제공한 원격 main SHA를 재확인하거나 원격을 갱신했다고 주장하지 않는다.
