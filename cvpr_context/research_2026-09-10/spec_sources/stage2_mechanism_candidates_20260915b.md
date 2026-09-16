# 2번 후속: MoFit 신호가 어느 조건에 묶이는지 확인하는 진단

2026-09-15. 새 응답을 보기 전 기록. 현재 환자 14393의 E2/U2, 두 target, 완료·검산된 원형 기반 MoFit 8건만 사용한다. 새 최적화·환자·학습·점수 선택은 없다. 아래는 아직 확인되지 않은 설명 후보이며, 기존 공격의 실패 원인으로 확정하지 않는다.

## 원문에서 확인되는 구조

- [X12 원문](../pdfs/X12.pdf) p.5–6, Fig.2와 Eq.(5)–(7): 각 **개별 query image**의 surrogate를 만들고, 그 surrogate에 embedding을 맞춘 뒤, 동일한 원본 이미지에 적용한다. 두 최적화와 평가에서 timestep과 sampled epsilon을 함께 고정한다. 이것은 image-specific membership의 설계이며, 학습에 없었던 동일 환자 사진 U에 환자 공유 신호가 보존된다는 주장은 별도로 입증되지 않는다.
- p.5는 원본 사진에 embedding을 직접 맞추면 member와 hold-out 모두 낮은 loss로 정렬되어 구별력이 줄어든다고 설명한다. p.9, Fig.4는 surrogate/embedding의 강한 적합 자체가 양쪽에서 일어남을 보여준다. 따라서 현재 최적화 loss가 잘 줄었다는 사실만으로 U의 참여 정보가 추출됐다고 할 수 없다.
- p.9, Table4에서 MS-COCO AUC는 clean-image embedding 85.59, random-perturbation 89.76, MoFit 94.17이고 TPR@1%FPR는 31.00, 29.20, 47.00이다. 원문의 성공 근거는 **그 조건에서 최적화한 surrogate의 효과**다. Random perturbation의 크기는 원문에서 데이터셋별 최고 결과로 선택했으므로 그대로 고정 비교 계약으로 옮길 수 없다.
- p.23–24, A.12/Table14는 Pokemon 100 member/100 hold-out에서 target epsilon을 달리한 네 **전체 방법 실행**을 비교한다. AUC 95.85–96.96%, TPR@1%FPR 18–41%이다. 이는 고정 embedding을 새 epsilon에 옮긴 실험이 아니다.
- [공개 COCO 코드](../code_sources/X12/COCO/MoFit_COCO.py): 175–177행 고정 epsilon 로드; 211–222행 surrogate 최적화; 329–339행 동일 epsilon으로 embedding 최적화; 339행 `optimizer.step()` 이후 344–355행 원본 사진 평가 및 conditional−unconditional score. 현재 동결 adapter의 `final.embedding` 역시 마지막 업데이트 **후** 실제 최종 h를 만든 좌표다 (`code_working/u_patient_audit/mofit_medical_adapter.py` 295–304, 334–339행).

## 후보 1: 자기 사진과 자기 모델에 맞춘 embedding의 효과가 분리되지 않았다

기존 M1 자기 embedding의 h와 M2 자기 embedding의 h를 빼면, query recipient 모델과 embedding을 만든 모델이 동시에 바뀐다. 따라서 그 차이 전체를 recipient 모델에 남은 학습 참여 반응으로 읽을 수 없다. 동일 환자 다른 사진으로 옮겼을 때 신호가 유지되는지도 아직 검사하지 않았다.

가장 작은 확인은 저장된 8개 embedding을 그대로 두고, 2개 recipient 모델 각각에서 네 query 사진에 전부 적용하는 것이다. 같은 사진에서는 2×2 표를 얻는다. A=h(M1,e1), B=h(M1,e2), C=h(M2,e1), D=h(M2,e2)로 두면 기존 A−D를 다음 두 경로 평균으로 정확히 분해할 수 있다.

`recipient 성분 = ((A−C)+(B−D))/2`

`embedding 성분 = ((A−B)+(C−D))/2`

둘의 합은 A−D이다. 이는 산술 분해이며 개인의 학습 포함/제외에 대한 인과 효과가 아니다. 모든 사진과 E/U mean 및 max에서 보고하되, max는 각 2×2 cell에서 먼저 집계한다.

자기 사진 대각선에서만 변화가 있고 같은 환자 다른 사진에서 사라진다면, **이 조건의 응답이 사진/조건 조합에 묶인다**는 좁은 관측을 얻는다. 다른 사진에서도 유지되면 “완전히 자기 사진에만 한정”된 설명은 약해진다. 어느 쪽도 환자 특이성을 확정하지 못한다. 다른 환자 대조가 없기 때문이다. 일반적 조건 불일치나 표준 관계 특징으로 설명될 수 있으며, 그런 경우 기존 baseline을 먼저 인정해야 한다.

## 후보 2: 최적화에 사용한 epsilon과의 결합이 응답을 지탱한다

원형은 Eq.(5), (6), (7)의 epsilon을 공유한다. 저장 embedding을 원래 epsilon과 사전 고정한 새 epsilon 네 개에서 평가하면, 현재 응답이 그 최적화 noise 밖에서도 보존되는지를 새 최적화 없이 분리할 수 있다.

새 epsilon에서도 target 대조와 교차 사진 패턴이 대체로 유지되면 “현재 현상이 오직 그 한 noise 적합 때문”이라는 설명이 약해진다. 사라지거나 방향이 달라지면 이 **고정 embedding 평가**가 noise 조건에 민감하다는 근거다. 다만 원형이 요구한 coupling을 의도적으로 바꾼 것이므로, 이것만으로 MoFit의 원형 MIA가 실패했다고 말하면 안 된다. Table14처럼 매 noise마다 재최적화한 재현도 아니다. 통상적 noise averaging 등 기존 방법으로 해결되는지는 별도이며, 그 자체를 새 기여로 부르지 않는다.

## 고정된 최소 연산 및 판정 범위

2 recipient × 4 query × (8 optimized embedding + 1 null) × 5 epsilon = **360 UNet forward, backward 0, VAE 0**. FP32, eval, batch1, t140. 새 epsilon은 CPU seeds 26091501–26091504, 원래 seed는 260915다. 모든 query/모델/조건에서 같은 noise를 써 비교의 Monte Carlo 변동을 줄인다. 실제 runtime은 아직 미측정이다.

각 사진의 저장 original posterior sample이 두 모델에서 exact 동일함을 CPU로 검사한다. 8개 원래 diagonal과 8개 null의 prediction/noised latent/loss/h를 재현하며 exact 여부와 오차를 모두 보고한다. 사전 FP32 실행 일치 허용치 rtol=1e-5, atol=1e-6은 실행 무결성용이며 효과 크기나 성공 기준이 아니다. 모든 4×4·2×2·5 noise 결과를 보존하고 좋은 것만 고르지 않는다. 원래 8건 결과는 바꾸지 않는다.

현재 CDI의 E→U 및 U 재학습 비교는 위 분해를 대체하지 않는다. 반대로 이 한 환자 MoFit 진단도 CDI의 모집단 성능이나 환자 MIA의 가능한 최고 성능을 판단하지 못한다. 이번 결과는 다음 원인 조사에서 무엇을 더 이상 단정할 수 없는지 좁히는 자료다.
