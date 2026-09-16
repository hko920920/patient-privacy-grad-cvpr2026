# t140 반복 noise 진단의 근거와 고정 범위

현재 연구 2번. 새로운 attack 설계나 원인의 확정이 아니라, 기존 generic/t140 loss 측정이 한 noise draw에 민감했는지 확인하는 제한된 진단이다.

## 실제 코드·저장 tensor에서 확인한 사실

| 기존 경로 | 현재 실제 noise 처리 | 이번 진단이 대체하지 않는 부분 |
|---|---|---|
| CDI Denoising Loss | t100에서 독립5개 noise의 L2 평균. 저장 raw에[1,5,1] 값과 각각의 prediction/noise가 존재 | 이미 noise 평균을 한 baseline이며, 전체가 single-noise였다고 말하면 틀림 |
| CDI Multiple Loss | noise1개를 t0,100,…900에 공유. n_repetitions=5 설정은 공개 클래스에서 사용하지 않음 | t140 반복만으로 열 timesteps의 공분산·전체26D learned scorer 원인을 판정하지 못함 |
| CDI GM | noise1개를 열 timestep의 mask/response에 공유 | gradient/mask의 noise 민감도는 이번 forward-only loss 진단 밖 |
| CDI NO | noise1개 고정으로 L-BFGS objective를 여러 번 평가 | optimizer 평가 횟수는 독립 noise Monte Carlo 표본 수가 아님 |
| SecMI·PIA/PIAN | 이 구현에서 외부 random noise draw0; deterministic prediction 경로 | 이 baseline들의 약한 결과까지 random-noise 탓으로 설명할 수 없음 |
| 기존 prompt/t140 | 영상당 CPU Gaussian1개를 base/두 target/세 prompt에 공유 | 이번에는 generic·두 target만 반복, base나 다른 prompt는 반복하지 않음 |

직접 확인: code_sources/X01/src/attacks/features_extraction의 denoising_loss.py:22–33, multiple_loss.py:15–29, noise_optim.py:45–82, secmi.py, pia.py; code_working/u_patient_audit/cdi_adapter.py:74–94,173–178,339–343. 실제 U selection raw의 module_outputs.denoising_loss shape와 noise_draws 수5/0/0/1/1/1도 CPU에서 확인했다. 전체5-draw 통계는 이 준비 작업에서 계산하지 않았다.

## 기존 방법이라는 근거

- CDI 원문 X01 p4 §3.1은 DL을 t100에서5회 계산해 평균한다고 명시한다. 반복 noise 평균 자체는 이미 비교군의 일부다.
- MoFit X12 p23 Appendix A.12/Table14는 Pokemon100 member+100 holdout에서 target noise4개를 바꾼 결과를 비교한다. AUC95.85–96.96이고 TPR@1%는18–41로 변한다. 이는 해당 설정의 실험이며 현재 의료 모델의 noise 안정성 보증이 아니다.
- PFAMI X11 p6 §IV-D는 sampled timestep의 estimation error 평균을 사용한다. timestep 평균과 동일 timestep에서 독립 noise를 반복하는 평균은 구별해야 한다.

따라서 단순 noise 평균으로 성능·안정성이 개선되면 기존 표준 측정의 보완 근거이지 새로운 patient-structure 기여가 아니다.

## 승인된 실제 비교

selection40(20A/20B), 환자별 E2/U2, 기존 두 target에서 t140 generic condition을 고정한다. 원래 draw0의 prediction·MSE를 정확히 재사용하고 신규7개의 독립 noise만 더한다. 같은 영상/repeat의 noise는 두 target에 공유한다. Noise seed는 image/repeat 문자열만으로 정하고 membership label을 사용하지 않는다.

- 160개 고유 영상×2target×8draw=2,560개 저장 cell.
- 기존320개 재사용, 신규2,240F/0B. VAE/text encoder 및 optimizer 실행0.
- 기존4,320F 약400초 실측을 선형 환산하면 새 forward는 약207초다. 모델 적재·검산·기록은 별도여서 실제 실행 시간을 보고한다.
- MSE는 기존 FP32 scalar와 동일한 연산; L2는 모든 draw에서 저장 FP32 prediction의 CPU FP64 residual norm으로 일관되게 계산한다.
- 각 seed8개와 prefix 평균1/2/4/8 전부 보고. 각 영상의 noise 평균→고정 negative 부호→환자 mean/max 순서다.
- 기존2,000개 patient bootstrap index를 재사용하여 두 target/조건/seed/contrast에 공통 적용한다. 192개 AUC표와48개 평균1 대비 contrast를 모두 보존하며 승자·seed·방향을 고르지 않는다.
- 반복 내 score 분산, 28개 seed-pair Spearman, 서로 겹치지 않는1/2/4-draw block 평균 간 순위, 동일 noise의 참여−비참여 target 차이를 표시한다.
- Single-draw patient max를 noise에 걸쳐 평균한 값과 noise 평균 후 patient max를 구분한다. ICC는 전자의 반복 측정 기술량이며 참여특이성 계수가 아니다.

구간은 이8개 noise 실현값과 기존 개발 환자에 조건부다. 두 target은 전체 A/B 학습군이 달라지는 구조여서 개인 인과효과가 아니다. 반복 결과가 양성이어도 full CDI/MoFit의 실패 원인을 확정하지 않고, 음성이어도 누출이 없다는 뜻으로 해석하지 않는다. 20negative에서1%FPR 성능이나2번 전체 합격선을 만들지 않는다.

실행 파일: run_repeat_noise_t140.py. 독립 저장 산술/통계 대조 파일: analyze_verify_repeat_noise_t140.py. 기존 동결 코드·raw는 수정하지 않는다. CPU FP64 검산은 저장 예측의 재계산이며 독립 neural forward 재현이 아니다.

