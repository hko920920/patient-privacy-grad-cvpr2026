# 방향1: 생성 연결은 통과했지만 현재 생성 결과는 부적합

**연구적 판정: 현재 구성의 결과는 나쁘다.** 정확히 연결된 작은 보정층이 이번 고정 두 조건에서 흉부 X-ray 형태를 만들지 못했다. 기반 모델·공개전용·공개+사적의 **6장 모두 생성 목적에 부적합**했다. 구현 검산 PASS를 연구 성공으로 해석하지 않는다. **96장 확대와 새 DP solver 탐색을 보류하고, 기존 흉부영상 생성 양성 대조로 출발 조건부터 확인한다.**

현재 큰 단계2·방향1의 [현실 계획](REALISTIC_RESEARCH_PLAN_20260916.md) §5B 중 연결 검증·최소 미리보기까지 완료했다. [실행 전 명세](TRACK1_SAMPLING_INTEGRATION_PROTOCOL_20260916.md)의 판정 규칙을 적용했으며 조건·seed·가중치를 결과에 맞춰 바꾸지 않았다.

## 1. 무엇을 실행했고 무엇이 나왔는가

| 판단 대상 | 결과 | 의미 |
|---|---|---|
| 보정 연결의 정확성 | **좋음: 독립 검산6,475개 PASS** | 원형 학습 특징·시간 basis·보정식·DDIM 경로가 지정한 수치 조건과 일치 |
| 현재 모델의 흉부영상 생성 적용성 | **나쁨: 고정6장 모두 부적합** | 현재 설정으로 96장을 확대하거나 보호 solver를 바꾸는 근거가 부족 |
| 사적자료의 추가 생성 효용 | **미확인** | 공개/pooled가 모두 목적 도메인을 형성하지 못하므로 미세한 우열 평가를 하지 않음 |
| 환자DP 구성의 가치·논문 기여 | **미확인** | 이번은 비DP3조건의 연결 확인이며 DP 영상을 생성하지 않음 |

두 prompt는 기존 weak-label no-finding과 pleural-effusion 문구다. 각 prompt에 미리 정한 CPU 초기 noise seed 하나를 쓰고, base/zero/public/pooled/base_restored 다섯 경로를 같은 입력으로 실행했다. DDIM30·256px·guidance1·eta0, UNet FP32를 유지했다. 검산을 통과한 뒤 base/public/pooled의 최종 latent만6장으로 decode했다.

![미리 고정한 6장 전체: 위 no-finding, 아래 effusion; 각 행 base/public/pooled](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/preview/comparison_grid.png)

no-finding의 세 영상은 흑백 줄무늬·반복 패턴이고, effusion의 세 영상은 회색 배경의 분홍색 물체 형태다. 흉부영상의 기본 외형을 갖추지 못했다. Root와 별도 검토자가 같은6장 전체를 직접 확인했다. 의학적 병변 판독이나 임상 정확도 평가를 한 것은 아니다. 공개전용과 pooled는 육안상 유사하지만, 이 소표본으로 동등성을 주장하지 않는다.

## 2. 연결 검증의 실제 범위

- **공개 witness4개 재추론:** 원 latent·conditioning·noise seed로 noisy input을 복원했다. noise가 저장 target과 exact, 현재 기반 UNet 출력과 projected features도 네 witness 모두 과거 저장값과 exact였다. 최종 평가 환자 영상은 평가하지 않았다.
- **보정식:** conv_out 직전 특징을 원래 FP32 projection에 통과시키고, 상수1과 전체1000단계의 시간 basis를 같은 순서로 붙였다. FP64 residual을 FP32로 바꿔 epsilon 예측에 더했다. 독립 NumPy 전개로 이를 대조했다.
- **원상복구:** W=0과 연결 제거 후에는 두 prompt 모두30개 UNet 출력·모든 scheduler 입력·31개 latent가 원 기반 경로와 exact였다. witness에서 public 보정 반복 호출도 exact였다.
- **scheduler:** 실제 설정은 epsilon·DDIM30이며 timestep958→…→1이다. 300개 전이를 독립 CPU FP32 식과 FP64 수식으로 검산했다. CPU/GPU 전이는 bitwise exact가 아니라 사전 고정 tolerance와 연산 크기 기반 FP32 반올림 범위 안에서 일치했다.
- **변경·오염 방지:** UNet 전체 상태 hash가 전후 일치했고, gradient·남은 hook·역전파는0이었다. decode clamp 전에도 NaN/Inf가 없었다. preview 전에 독립 검산이 결속한 계약·manifest·trajectory hash를 다시 확인했다.

독립 검산기는 모든 UNet step을 별도로 GPU 재추론한 것이 아니다. 저장된 특징/출력/전이의 산술과 실제 재호출의 원상복구 증거를 검산했다. 연결 PASS는 전체 생성 설정의 적합성이나 의료 효용을 증명하지 않는다.

## 3. MSE가 줄었는데 왜 계속 진행하지 않는가

앞선 MSE는 **실제 흉부영상에 noise를 더한 입력에서 noise를 맞히는 오차**였다. 이번에는 순수 noise에서 시작해 끝까지 영상을 생성했다. 두 검증이 다르며, 현재 작은 head의 MSE 감소가 유용한 의료 생성으로 이어진다는 핵심 연결을 이번 예시에서는 확보하지 못했다.

보정이 적용되지 않은 것은 아니다. 두 조건의 public 보정은 기반 모델 대비 최종 latent를 변화시켰다. 사후 계산한 pooled−public latent RMS 차이도 no-finding0.05069/effusion0.02928로0이 아니다. **출력 변화가 있다는 사실은 품질 개선과 다르다.** [사후 변화 크기 기록](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/descriptive_changes.json)은 기술적 참고이며 품질 지표가 아니다.

현재 단정할 수 있는 것은 **일반 SD2.1+현재 작은 head+현재 sampling 조건의 부적합**이다. 의료 domain 사전학습 부족, head 표현력, guidance·precision·conditioning 등 조건의 영향을 인과적으로 분리한 것은 아니다. 사적 데이터가 원래 불필요하다거나 환자DP 연구 전체가 불가능하다고 결론내리지 않는다.

## 4. 다음 한 단계: 이미 있는 생성 양성 대조와 설정 연결

새 checkpoint를 무작정 찾기 전에 활용할 대조가 이미 있다. [기존 step1000 LoRA 비교 영상](../../code_working/_reports/cvpr_u_pilot_v1_001/quality/training_coverage_v2/step_1000_both/comparison_grid.png)에서 M1/M2는 기본적인 정면 흉부영상 외형을 만들었고, base는 당시에도 비의료적 출력을 냈다. [기존 판독 기록](../../code_working/_reports/cvpr_u_pilot_v1_001/quality/visual_review_pair.json)의 PASS는 기본 형태에 한정되며 질환·임상 효용의 PASS가 아니다.

기존 대조는256px·DDIM30으로 해상도·step은 같지만 **FP16+autocast·CFG7.5·CUDA 초기 noise·pipeline 직접 text encoding**이었다. 현재는 FP32·CFG1·CPU 초기 noise·저장 conditioning이다. 같은 seed 숫자를 복사해도 CPU/CUDA RNG가 다르면 같은 초기 latent가 아니다. 과거 base도 CFG7.5에서 부적합했으므로 guidance1만 원인이라고 추측해 결론내리지 않는다.

다음은 **기존 LoRA의 알려진 생성 경로와 현재 경로 사이의 차이를 고정된 입력으로 확인하는 최소 양성 대조**다. 예상 설계·구현·검산 **30–60분**, 계산은 기존 실측상 수분 이내가 예상되며 실행 명세를 만든 뒤 갱신한다. 새로운 보호법을 학습하거나 공개/사적 가중치를 조정하지 않는다. 의료 생성 출발점과 최종 보호 설계를 구분한다.

M1/M2는 사적 역할 자료로 학습한 모델이므로 **양성 대조로만 사용한다. 환자DP가 보장되는 공개 backbone으로 자동 채택하지 않는다.** 그 가중치 위에서 head에만 DP를 적용해 전체 모델이 환자DP라고 주장할 수 없다. 보호 연구를 이어갈 기반 모델은 공개·사적 자료 경계와 보호 범위를 별도로 충족해야 한다.

## 5. 비용과 보존된 산출물

| 항목 | 실제 결과 |
|---|---:|
| UNet forward / backward | 325 / 0 |
| 본 실행(모델 로딩·fingerprint·저장 포함, Python import/사전 hash 검증 제외) | 36.285초 |
| 10개 sampling 경로 합계 | 18.427초 |
| 30step 경로당 시간 | 1.779–2.007초 |
| 최대 PyTorch allocated GPU memory | 약3.588GiB |
| 독립 CPU 검산 | 4.499초·6,475개 |
| 추가 VAE decode | 6회, preview6장 |

이 시간은 고정된2조건의 연결 profile이며 DP-LoRA 대비 효율 우위를 측정한 것이 아니다. 전체 작업 예상45–90분에는 구현·원문/코드 대조·검산·판독·기록이 포함된다. 96장 및 RAD-DINO/BioViL-T 품질 평가는 실행하지 않았다.

[계약](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/contract.json) · [runtime](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/runtime.json) · [실행 기록](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/execution.json) · [전체 경로 manifest](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/manifest.json) · [독립 검산](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/independent_verification.json) · [6장 manifest](../../code_working/_reports/frozen_residual_sampling_integration_20260916_v1/preview/manifest.json).

실행 기록의 `COMPLETED_PENDING_INDEPENDENT_VERIFICATION`은 생산 종료 당시 상태로 보존했다. 후속 독립 검산 파일이 PASS이고 preview도 그 뒤에 실행됐다. 현재 진행 상태는 두 연구 state와 본 보고서를 따른다. 이 환경에서는 local git commit이 조회되지 않아 manifest의 source_commit은null이고 실제 입력·코드는 SHA256으로 결속했다. 사용자가 제시한 원격 커밋 확인을 여기서 재실행했다고 주장하지 않는다.
