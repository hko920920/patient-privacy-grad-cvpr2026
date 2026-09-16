# 1번 안의 PFAMI 비교 — 고정 실행 계약
2026-09-14. **큰 순서의 1번, 기존 방법이 잡는 정보와 놓치는 정보를 확인하는 작업이다.** 앞선 원문·코드 검토 다음에 수행하는 비교이며, 새 R 방법의 확대 단계가 아니다.

## 이번에 답할 질문
기존 두 의료 LoRA 모델의 U 관측 영상에서 **원본/crop의 상대 손실 변화**가 단순 손실 차이보다 환자 판별 정보를 더 주는가? E는 같은 실행의 진단 대조다. 이것이 SecMI가 약했던 원인을 확정하거나 R의 기여를 증명하는 실험은 아니다.

[원문 대조](METHOD_PREMISE_AUDIT_2026-09-14.md)의 PFAMI를 선택했다. 원저자 통계의 loss/feature 계산을 보존하고, 256px 의료 latent diffusion에 적용한 PFAMI-style adaptation으로 명시한다. 원 논문의 SD 성능을 재현했다고 부르지 않는다.

## 실행 전에 고정한 선택
- 기존 selection40명, 환자당 E 2장·U 2장, model_1/model_2. fit 환자14393의 별도 실행은 전처리·시간 확인용이며 성능 표에서 제외한다.
- 이미 공개한 일반 prompt와 실제 epsilon scheduler를 유지한다.
- 원본은 학습 때 만든 FP16 VAE posterior-mode cache를 쓴다. crop은 canonical256px 영상의 tensor 중앙207px를 bilinear/antialias=True로256px로 되돌린 뒤 같은 FP16 VAE mode로 인코딩한다. UNet·loss는 FP32다.
- crop 강도는 원저자 stat의 grid index5, 73/90이다. timestep은0,50,…,450, 각원본/crop당noise1개다.
- 원본/crop, timestep, target/base 사이에는 독립 noise를 쓴다. 두 target 모델의 동일 영상·view·시점만 noise를 공유하여 모델 비교의 Monte Carlo 변동을 줄인다.
- 주점수: `mean_t[(L_crop-L_original)/L_original]`, 큰값=member. 그후 같은환자영상2개의 평균을 낸다.
- 주비교: `mean_t[L_crop-L_original]`. 같은20forward를 사용하므로 정규화의 차이를 비교할 수 있다. 이는 후속SD 구현의 점수 형태를 같은입력에 적용한 대조이며 그논문전체의 재현은 아니다.
- 음의 원본loss 평균은 부가 진단이다. 독립실행하면10forward이므로 같은비용의우위라고 말하지 않는다.
- base의 상대반응을 차감한 값은 부가진단이다. 공개base는 원문의 별도disjoint의료reference모델이 아니다. 보조점수로U주비교를 구제하지 않는다.
- 분모 epsilon이나clipping을 넣지 않는다. 0·비유한loss는 실행오류로기록하고 원인을검토한다.

## 표본·분석·종료
selection40은 기존개발에서 이미 본자료이며 미개봉시험이아니다. 환자A/B각20명을층화한2,000회bootstrap을 모든점수·모델·E/U에서공유한다. AUC수준과정규화−차이의paired ΔAUC및95%구간을모델별보고한다. 시점·사진을독립환자로세지않는다.

새로운 임의합격선을 만들지 않는다. 양쪽U에서상대이득이관측되더라도 개발관측으로제한하고절대판별력도함께본다. 구간이넓거나방향이다르면추가이득미확인이다. 어떤결과든이번고정측정후닫으며 부호·crop·시점수정,표본확대,R튜닝,추가target학습,calibration/test개방을자동으로진행하지않는다.

## 계산량과 기록
본측정은160개고유영상,320target영상평가+160base영상평가다. target6,400F+base3,200F=9,600F/0B이며 crop VAE160영상의인코딩비용은별도다. fit1벤치마크도별도계상한다. 원시20개잔차와시점별loss를보존해CPU에서손실·정규화·환자집계·통계를독립검산한다.

기계실행계약: `code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_preparation_v1/contract.json`. 실제실행protocol은입력·체크포인트·계산코드hash를결속한다. 기존결과를덮어쓰지않는다.

시간안내: 정의·연결검토15–25분을먼저보고했다. GPU시간은fit1실측후별도보고하며문헌query수를확정walltime으로환산하지않는다. 이문서는실행전계약이고 실제결과는후속실행기록을따른다.

