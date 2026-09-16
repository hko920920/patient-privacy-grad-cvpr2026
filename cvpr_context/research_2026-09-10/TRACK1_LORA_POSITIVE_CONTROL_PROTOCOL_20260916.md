# 방향1: 기존 LoRA 양성 대조 재현과 sampling 조건 연결

**실행 전 판단:** 일반 SD2.1+작은 head의 앞선 생성 결과는 나빴다. 이번에는 이미 기본 흉부영상 형태를 만든 기존 모델로 생성 출발점과 경로를 확인한다. 성공해도 새 head의 효용이나 환자DP 기여가 확인되는 것은 아니다.

M1 step1000을 이름 순서로 고정한다. 옛 no-finding/effusion 두 이미지, 원 prompt, 원 CUDA seed와 체크포인트를 사용한다. M2나 보기 좋은 seed를 추가 선택하지 않는다. 기존 generic-base head는 **적용하지 않는다**. M1은 사적 역할 자료로 학습한 진단용 대조이며, 공개DP-safe backbone이나 최종 독립 효용 평가 모델이 아니다.

## A. 과거 결과부터 재현

원 loader대로 backbone FP16·LoRA FP32·autocast FP16을 보존한다. VAE FP16, pipeline 직접 text encoding,256px,DDIM30,CFG7.5,CUDA noise generator를 유지한다. 출력 PNG는 원래와 같은 uint8 버림으로 저장한다. 원 historical initial latent는 저장돼 있지 않아 동일한 초기값을 미리 보장하지는 않는다. 이번 실제 초기 latent·embedding·UNet 출력·guided epsilon·scheduler 전이·decode를 전부 기록하고 과거 PNG와 파일/픽셀 exact 일치를 목표로 대조한다.

원 함수 wrapper는 반환값을 바꾸지 않고 `finally`에서 복구한다. scheduler.step wrapper는 signature를 보존한다. A를 별도 실행·독립 검산하고2장 모두 직접 확인한 뒤 B로 간다. 재현이 맞지 않으면 기록을 보존하고 원 조건부터 검토한다.

## B. 실제로 캡처한 같은 입력으로 경로만 변경

원 pipeline에서 나온 초기 latent·[null,conditional] embedding을 그대로 직접 UNet→DDIM 루프에 넣는다. 같은 mixed weights·autocast·CFG7.5·scheduler·VAE를 유지한다. 30step 전체 입력·raw epsilon·CFG epsilon·latent와 최종 decode/image가 원 pipeline과 exact인지 확인한다. CFG는 같은 batch2에서 원 dtype과 순서를 유지한다.

## C. 고정된 순서로 현재 조건까지 연결

| 단계 | 직전 단계에서 변경하는 것 |
|---|---|
| 원형 pipeline | 과거 조건 재현 |
| 직접 FP16 경로 | 경로만 변경, 실제 입력 동일 |
| 직접 FP32·CFG7.5 | UNet/VAE 및 입력을 같은 값의 FP32로 승격, autocast 해제 |
| 직접 FP32·CFG1 | guidance와 그에 따른 null branch 제외만 변경 |
| 직접 FP32·현재 conditioning | 기존 캐시의 conditional embedding만 사용 |
| 직접 FP32·현재 noise | 직전 sampling 패킷 base가 실제 사용한 초기 latent로 변경 |

FP32 승격 후 다시 half로 돌아가지 않는다. 같은값 FP32로 표현한 UNet 가중치 hash를 전후 비교해 원 가중치가 보존되는지 확인한다. 최종 단계의 초기 latent·conditioning은 앞선 base/head 패킷과 exact 대조한다. 이 순서는 해당 경로에서의 변화를 보여주며, 서로 상호작용하는 요인의 보편적 인과 분해라고 주장하지 않는다.

각 단계2조건, 총12경로·12장·360UNet 호출(540 batch examples),새역전파/학습0이다. 이미지·수치를 보고 조건을 조정하지 않는다. 모든12장을 보존하고 판독한다. 독립 검산은 저장 CFG/DDIM/영상 변환과 A/B 일치·조건 변경·hash를 확인하며 UNet/VAE 자체를 다시 CPU 추론하는 것은 아니다.

## 판단과 다음 범위

- 과거 경로 재현 실패: 원 checkpoint·환경·입력부터 복구한다.
- 원 경로는 성공하지만 어느 연결 단계에서 기본 외형이 무너짐: 그 조건의 적합성을 먼저 확인한다.
- 현재 동일 입력 조건에서도 LoRA가 기본 CXR 외형을 만들면: 일반 base+작은head의 출발 구성 문제 쪽으로 좁힐 근거다. 작은head 전체의 불가능성을 증명하지 않는다.
- 후속 후보는 공개 의료 기반모델 위에서 다시 특징을 추출·학습하는 작은 patient-private head이다. 기존W를LoRA위로옮겨그효과를판정하지않으며,새후보의추가효용·DP보존·품질비용우위는별도미검증이다.

처음 보고한 예상은 **30–60분의 설계·실행·검산·기록**, 실제 생성 계산은 수분 규모다. 이번에96장 품질평가·새solver·새backbone/head학습은 하지 않는다.

[과거 자료 감사](spec_sources/lora_positive_control_provenance_20260916.md) · [생산 코드](../../code_working/frozen_residual_head/run_lora_positive_control.py) · [독립 검산기](../../code_working/frozen_residual_head/verify_lora_positive_control.py).
