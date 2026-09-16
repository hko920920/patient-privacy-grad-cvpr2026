# 방향1: LoRA 재현과 현재 생성 조건 대조 결과

**판정: 원인 범위를 좁히는 진단에는 좋은 결과다. 생성 품질은 여전히 부족하고, 작은 head의 연구 효용은 미확인이다.** 과거 LoRA 이미지 두 장을 파일까지 정확히 재현했다. 같은 실제 입력을 직접 sampling 경로에 넣은 결과도 원 pipeline과 정확히 일치했다. 이후 현재의 FP32·CFG1·캐시 conditioning·초기 잡음까지 바꿔도 흉부 방사선영상의 기본 형태는 남았다. 다만 마지막 두 영상에는 뚜렷한 잡상과 형상 왜곡이 있고, 특히 effusion prompt의 영상은 중첩·회전된 구조가 심하다. 의료적으로 타당한 영상 생성 성공으로 판정하지 않는다.

**이번에 답한 질문은 ‘현재 sampling 조건에서도 학습된 의료 LoRA가 흉부 형태를 만들 수 있는가’다. 답은 이 두 고정 입력에서는 그렇다.** 따라서 앞선 일반 SD2.1+작은 head의 실패를 sampling 경로 전체의 고장만으로 설명하기는 어렵다. 기반모델의 도메인 적응, 작은 head의 표현력·학습목표 중 하나를 유일 원인으로 확정한 것은 아니다. 현재 큰 단계2·방향1을 유지한다.

## 실제 12장 전체

왼쪽부터 원형 재현 → 같은 입력의 직접 FP16 경로 → FP32·CFG7.5 → CFG1 → 현재 캐시 conditioning → 현재 초기 잡음이다. 위는 no-finding prompt, 아래는 effusion prompt다. prompt 이름은 요청 조건이며 질환 정확성 판독 결과가 아니다. 예정된 이미지 전부를 표시한다.

![고정 두 조건의 여섯 단계 전체](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/comparison_grid.png)

## 무엇이 확인됐는가

| 대조 | 실제 결과 | 허용되는 해석 |
|---|---|---|
| 원 M1 step1000, 원 prompt·CUDA seed·FP16/autocast·CFG7.5 | 두 PNG 파일 SHA와 픽셀이 과거 기록과 정확히 같음 | 알려진 양성 대조가 재현됨 |
| 원 pipeline → 직접 UNet/DDIM, 실제 초기 latent·embedding 동일 | 두 조건의 30step 입력·raw/CFG epsilon·latent·decode·image가 정확히 같음 | 직접 sampling 연산 경로가 원 pipeline을 재현함 |
| FP32 승격 | 기본 형태 유지, 수치는 달라짐 | FP32 조건이 이 LoRA의 흉부 형태 형성을 막지 않음 |
| CFG7.5 → 1 | 형태는 남지만 대비·구성이 크게 변함 | CFG1을 동일 품질 조건이라고 하지 않음 |
| 직접 embedding → 기존 캐시 | 형태 유지, 수치는 동일하지 않음 | 캐시 교체가 이 두 예시의 형태를 전면 붕괴시키지 않음 |
| 현재 초기 잡음으로 교체 | 흉부 형태는 남음. 두 장 모두 artifact, effusion은 심한 왜곡 | 현재 조건의 coarse 진단은 긍정적이지만 생성 품질 성공은 아님 |

최종 단계의 초기 latent 값·conditioning hash·DDIM timestep 및 수치 설정은 앞선 실패 base 실행과 정확히 맞췄다. 같은 SD2.1 기반·VAE·입력 조건에 학습된 rank8 LoRA가 추가된 비교다. 서로 같은 모델이라고 하지 않는다. 앞선 base PNG는 반올림, 원형 LoRA PNG는 버림으로 저장하므로 화면 비교에 최대 1/255 수준의 표시 차이가 포함될 수 있다. 이것으로 흉부 형태와 기존 무늬/물체 사이의 차이를 설명할 수는 없다.

캐시 교체 전후 embedding 최대 절대차는 두 조건 모두 0.0234375였다. 최종 latent RMS 차이는 normal 약0.000252, effusion 약0.000365다. 시각적으로 비슷하다는 관찰과 수치적으로 같다는 주장을 구분한다. 이 값은 품질 지표가 아니다. 순차 변경은 앞선 조건에 의존하며 독립 요인의 보편적 인과 효과를 추정한 실험이 아니다.

## 검산과 실행량

- 원형 재현 검산은 수정본 v2에서 **855개 PASS**. 최종 전체 패킷 검산은 **4,503개 PASS**, 12경로의 360 DDIM 전이를 포함한다. 두 검산의 중복 항목을 더해 독립 증거 수를 부풀리지 않는다.
- 전체 UNet 호출360회, batch example540개, VAE decode12회. 새 역전파·학습·head 적용0회.
- 원형 실행22.093초, 조건 대조43.748초로 두 실행기 합계65.841초. 12경로의 실제 생성 구간 합계40.019초. 최대 GPU allocated memory4,220,371,456bytes, 약3.93GiB. RTX3070에서 측정했다. 코드 작성·감사·기록 시간과 GPU 연산 시간을 구분한다.
- 원형과 대조 전후 UNet을 같은 FP32 값으로 표현한 hash가 동일했다. 기존 FP32 LoRA를 half로 재양자화하지 않았다. 새 gradient나 남은 hook도 없다.
- 검산은 저장된 입력·출력의 CFG/DDIM/영상 변환 및 hash를 독립 계산한다. UNet·VAE 전체를 별도 CPU 모델로 다시 추론한 검사는 아니다. GPU와 CPU DDIM의 모든 비트가 같다는 주장도 아니며 사전 고정한 dtype별 roundoff bound를 적용했다.

**최초 검산 실패도 보존했다.** 첫 검산기는 timestep을 int64로만 가정했지만 원형 캡처는 이 Windows 환경에서 int32로 저장됐다. 시간값은 같았다. 기존 결과·계약·코드를 수정하지 않고 timestep만 signed int32/int64로 검증하는 새 검산기와 별도 bridge 진입점을 만들었다. CFG/DDIM 계산·수치 허용범위는 바꾸지 않았고 원형 생성도 다시 하지 않았다. 정정 시점은 원형 replay 후·bridge 전이다. [정정 내역](spec_sources/lora_verifier_amendment_20260916.md)과 원 실패 파일을 함께 남겼다.

## 연구 판단과 다음 한 단계

앞선 **일반 SD2.1+현재 작은 head 설정은 사전 고정한 두 생성 예시에서 최소 형태 확인을 통과하지 못했다**는 판정을 유지한다. 이번 양성 대조는 그 head를 살린 결과가 아니다. 다만 현재 경로에서도 의료 적응 모델은 기본 흉부 형태를 만들 수 있으므로, 다음 후보를 **공개 의료 기반모델 + 그 위에서 새로 학습한 작은 patient-private 적응층**으로 설계할 근거는 강화됐다.

기존 M1은 사적 역할 자료로 학습한 비DP 모델이다. 현재 private head 학습80명 중40명, 개발40명 중20명과도 겹친다. 이것을 공개 DP-safe 기반모델이나 독립 최종 평가 모델로 재분류하지 않는다. 이번은 새 잠긴 calibration/test 환자 평가도 아니다. [학습 자료 출처 감사](spec_sources/lora_positive_control_provenance_20260916.md)에 범위를 남겼다.

다음 한 단계는 **공개 경계가 분명한 의료 기반모델의 확보 방법과 재학습 대조를 고정하는 것**이다. 로컬에서 재사용 가능한 공개 역할 의료 LoRA는 아직 확인되지 않았고, 공개32명만으로 새 적응을 할 경우 기존1000step을 그대로 복제하면 반복 노출량이 크게 달라진다. 공개 환자 수·훈련량·품질 기준·전체 비용을 먼저 맞춘다. 설계·자료 확인은30–45분 예상이며, 새 학습의 시간은 선정한 자료와 step 수를 근거로 별도 보고한다. 이번에는 새 공개 LoRA나 작은 head를 학습하지 않았다.

이후 그 기반에서 특징과 head를 **새로** 만들고 공개전용 대비 사적 추가 생성 효용을 확인해야 한다. 기존 generic-base용 projection과 W를 그대로 이식해 효용을 판정하지 않는다. 96장 자동 확대와 새 DP solver 탐색은 계속 보류한다. 임상 품질, 사적 추가 가치, patient-DP 아래 보존, 강한 기준선 대비 품질–전체 비용, CVPR 기여는 아직 확보하지 못했다.

## 재현 근거

- [실행 전 명세](TRACK1_LORA_POSITIVE_CONTROL_PROTOCOL_20260916.md), [원 계약](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/contract.json), [v2 정정 결속](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/verification_amendment_v2.json).
- [원형 실행](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/execution_replay.json), [조건 대조 실행](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/execution_bridge.json), [원형 검산 v2](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/replay_verification_v2.json), [최종 검산 v2](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/independent_verification_v2.json).
- [보존한 최초 검산 실패](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/replay_verification.json), [변경별 기술 통계](../../code_working/_reports/frozen_residual_lora_positive_control_20260916_v1/descriptive_contrasts.json).
- [원 생산 코드](../../code_working/frozen_residual_head/run_lora_positive_control.py), [수치 동일 v2 bridge 진입점](../../code_working/frozen_residual_head/run_lora_positive_control_v2.py), [독립 검산 v2](../../code_working/frozen_residual_head/verify_lora_positive_control_v2.py).

원 계약 SHA256: `b6d5b775c2bb0b63180d7143a0ef5de4b20217f6ce060e2d484a8fe59671457c`.
