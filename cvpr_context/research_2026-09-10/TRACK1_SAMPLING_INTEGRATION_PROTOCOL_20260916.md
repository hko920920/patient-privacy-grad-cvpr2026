# 방향1: 실제 sampling 연결 검증 명세

**실행 전 판단: pooled MSE 결과는 약한 연구 근거다. 이번 작업의 목적은 현재 보정층을 정확히 적용했을 때 실제 생성 경로에서 무엇이 바뀌는지 확인하는 것이다.** 구현 PASS와 유용한 의료 생성의 성공을 구분한다. [현실 계획](REALISTIC_RESEARCH_PLAN_20260916.md) §5B의 제한된 첫 작업이며 새 solver나 새 DP 학습을 하지 않는다.

## 고정 범위

동일 SD2.1 snapshot의 FP16 artifact를 FP32로 올린 직접 UNet에 full64 출력 보정을 붙인다. conv_out 직전320채널을 저장 P로 FP32 투영하고, 상수1·기존1000단계 logSNR의 네 Legendre 항을 원래 순서로 결합한다. FP64 W로 계산한 residual을 FP32로 바꿔 **epsilon 예측에 더한 뒤** DDIM에 전달한다. 가중치·시간 basis·projection은 기존 검산된 값을 그대로 사용한다.

실제 scheduler 설정은 epsilon, DDIM30·eta0·256px·guidance1로 고정한다. 현재 설정의 timestep은958→925→…→34→1이며 alpha와 logSNR 정규화는 학습 당시1000개 전체를 따른다. 새 null branch나 CFG 규칙을 도입하지 않는다. 기존 weak-label prompt의 저장 conditioning을 쓴다.

## 검증할 연결

1. 공개 witness0/7/16/23에서 원 latent·prompt·seed로 noisy input을 복원한다. 저장 noise와 현재 복원 noise, 원 특징/기반 예측, 시간 basis, offline/online residual을 대조한다.
2. 같은 입력의 W=0 출력, 보정 연결 제거 후 기반 출력, 같은 public 보정의 반복 호출은 exact equality를 요구한다.
3. 미리 정한 no-finding/effusion 두 prompt와 CPU seed26091631/26091632 각각에서 base/zero/public/pooled/base_restored 경로를 실행한다. 매 경로 자신의 현재 latent에서 특징을 다시 계산한다.
4. 10개 경로×30step의 모든 epsilon·latent를 저장한다. 별도 CPU 검산기가 보정식과 DDIM 전이를 다시 계산하고, zero/base/restored 전체 경로 일치를 확인한다.
5. UNet 전체 상태의 실행 전후 hash, gradient 부재, hook 해제, finite 값, 단계별 비용·메모리를 기록한다. 독립 검산기는 모든 UNet step을 다시 GPU 추론하는 검증은 아니다.

실행량은 warmup1 + witness4×6 + trajectory10×30 = **325 UNet forward,0 backward**다. FP32 반올림과 원 저장 재현 허용오차는 실행 전 JSON 계약과 검산 코드에 동결한다. 독립 검산기는 생산 수치 함수를 import하지 않는다.

## 최소 영상 확인과 판정

패킷 검산 PASS 뒤에만 두 고정 조건의 base/public/pooled 최종 latent를 VAE로 decode해 **6장 전부** 보존·확인한다. seed나 보기 좋은 이미지의 재선택은 없다. clamp 전에 raw decode의 finite 여부를 확인한다. VAE의 실제 scaling factor를 읽고 최종 latent를 그 값으로 나눈다.

- 구현이 틀리면 연결 문제를 고친다. 성능 결과로 해석하지 않는다.
- 올바른 경로에서도 현재 예시가 비의료적이거나 심하게 부적합하면, 96장으로 자동 확대하지 않는다. 현재 기반 모델·작은 head의 생성 적용성을 먼저 재검토한다.
- 기본 도메인 형성이 가능하고 방법 차이가 관찰되면, 고정 여섯 방법·96장과 기존 평가기를 연결할 근거로 사용한다. 이 작은 관찰만으로 public 대비 사적 생성 효용을 확정하지 않는다.

최초 총작업 예상45–90분. 실제 GPU profile 뒤 계산시간을 따로 보고한다. 최종 보고는 **연구적으로 좋은지/약한지/불명확한지 → 구현 결과 → 그 판단의 근거 → 다음 결정** 순서로 쓴다.

[원 소스 감사](spec_sources/sampling_integration_source_audit_20260916.md) · [공개 witness 감사](spec_sources/sampling_witness_audit_20260916.md) · [생산 코드](../../code_working/frozen_residual_head/run_sampling_integration.py) · [보정 연결](../../code_working/frozen_residual_head/sampling_adapter.py) · [독립 검산기](../../code_working/frozen_residual_head/verify_sampling_integration.py).
