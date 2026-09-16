# 방향1의 첫 실제 확인: 작은 동결 잔차층의 의료영상 적응 능력

2026-09-16. 사용자는 기존 검토의 도달점부터 보고하고 방향1을 먼저 실행하라고 지시했다. 현재2번의 설계 검토 안에서 후보의 핵심 연결 하나를 실제 자료로 확인한다. 이 파일은 실행 중 작성한 읽기용 설명이며, 결과 전 고정한 기계 명세는 `code_working/_reports/frozen_residual_capacity_20260916_v1/contract.json`이다.

## 확인할 질문과 큰 연구 목표의 연결

큰 목표는 환자 단위 보호 아래 생성 학습의 품질–비용 절충을 개선하는 것이다. 현재 후보는 큰 공개 denoiser를 고정하고 작은 공간·시간 잔차층을 환자별 보호 통계로 학습한다. 이 후보가 성립하려면 먼저 **그 작은 함수족이 의료영상의 적응 신호를 담아야 한다.** 이번에는 DP 잡음·clipping을 넣기 전에 이 부분을 확인한다.

DP를 넣지 않은 이번 head는 배포용 보호 모델이 아니다. 이번 결과로 생성영상 품질, 환자 MIA, 같은 head의 DP-SGD 대비 이득, 기존 DP-LoRA 대비 우월을 주장하지 않는다. 반대로 한 고정 projection·ridge 조건의 작은 효과나 실패를 보호 효율 연구 전체의 불가능으로 확대하지 않는다.

## 실행 전에 고정한 것

| 항목 | 고정 내용 |
|---|---|
| 기반 모델 | Manojb/stable-diffusion-2-1-base, revision 0094d483a120f3f33dafbd187ea4aa60d10de75c |
| 모델 로딩 | adapter 없는 원래 UNet. 저장 FP16 가중치를 FP32로 승격, eval/no-grad, TF32 off |
| 사진 | 기존 NIH 흉부 X-ray 개발자료. fit80명×4장=320장 학습, selection40명×4장=160장 개발평가 |
| 환자 분리 | 학습/평가 환자 교집합0. calibration140/test140은 이번 추출·선택에서 제외 |
| 이전 E/U | 출처 주석으로만 보존. 새 head는 학습80명의 기존4장을 모두 사용하므로 과거 E/U를 새 참여 라벨로 재사용하지 않음 |
| 전처리 | 기존 full-field 256 영상·FP16 VAE posterior-mode latent 캐시를 그대로 사용 |
| 조건 | 영상별 기존 weak-label prompt. 생성 때도 지정 가능한 질환 조건이며, 환자의 질환 label 자체가 공개 정보라는 주장은 아님 |
| 특징 위치 | UNet conv_out 입력, 마지막 normalization·activation 이후320채널 |
| 공간 특징 | 결과와 독립인 고정 Gaussian-QR 투영15채널 + 상수1 =16채널 |
| 시간 특징 | 공개 scheduler 전체의 log-SNR 범위로 정규화한 Legendre P0–P3 |
| 주 head | 공간16×시간4=64차원 → 출력4채널, 256파라미터. 모든 공간 위치에 공유 |
| 대안 | 같은 공간 특징의 시간 독립 head16, 공간 특징 없이 시간별 offset만 주는 head4, 원래 base |
| 회귀 | 절대 ridge λ=.001 한 값. 평가 결과로 차원·λ·projection·시점을 선택하지 않음 |
| 시점·잡음 | 각 영상마다 timestep을8개 구간으로 나눠 hash로 고정한 한 시점씩 선택. CPU torch generator의 독립 FP32 Gaussian noise |
| 자료수 | 480장×8=3,840개 noisy-input 기록. 픽셀이나 noise를 독립 환자로 세지 않음 |
| 목적 | 환자·환자 내 사진·사진 내 draw 균등 평균. 통계 Q는 출력4채널 제곱합, 보고 MSE는 Q/4 |
| 주 비교 | 새로운 환자40명의 base MSE − full64 head MSE |
| 보조 비교 | static16 − full64, time4 − full64. 환자 paired bootstrap8,000회, 기술적 개발평가 구간 |

각 환자의 A/B/Q를 먼저 만들고 학습80명만으로 회귀 해를 구한다. Q는 절대 loss를 재계산하기 위한 보호되지 않은 분석 통계이며 DP release에 포함하지 않는다. 공간16차원 통계의 Kronecker 확장으로64차원을 효율적으로 만들고, 독립 검산은 실제64열 행렬을 전개해 다시 계산한다.

## 비교의 해석

full64가 base보다 나아지면 이 고정 함수족이 일부 적응 신호를 담았다는 근거가 된다. static16과 time4도 좋아지면 그 표준 대안을 인정한다. full64만의 추가 이득은 이 두 비교와 분리해서 설명한다. Training에서만 좋아지고 환자-disjoint development에서 악화되면 일반화 근거가 부족한 것이다.

기존 LoRA 두 모델은 각912장·다른 환자 구성을 학습했다. 이번320장과 같은 학습 조건이 아니므로 그대로 공정한 비교 성능표에 넣지 않는다. 다음 보호 학습 비교에서는 **같은 head의 user-DP-SGD도 backbone backward0**이라는 점을 반영해야 한다.

## 시간과 검산

실행 전 profile은8기록+warmup1=9 forward였다. RTX3070에서 평균0.08412초/기록, 계산량 투영323.03초, 최대 GPU allocated 약3.59GiB였다. 본추출3,840기록+warmup1=3,841 forward를 로딩·저장 포함6–8분으로 고지했다. 합계3,850F/0B이며 기존 VAE/text 캐시를 재사용한다. Profile에서 성능값을 보고하지 않았고, 조건은 profile 전 고정했다.

원본 입력·모델·코드의 SHA, 실물 영상480개, 캐시, 기존 cohort lock와 평가 CSV 일치 검사를 보존한다. 실행 전 문구 수정(public prompt → cached weak-label prompt)은 초기 CPU 명세를 별도 보존하고 GPU 전에 새 코드 해시를 결속했다. 표본·방법·seed는 바꾸지 않았다.

별도 CPU 검산기는 생산자의 수치 함수를 import하지 않고 projection witness4개, 모든 basis/정답 차감/A/B/Q, 회귀 해·stationarity, 실제 잔차의 환자 MSE와 bootstrap을 확인한다. 이것은 저장 결과의 산술·출처 검산이며 전체 UNet forward를 별도 구현으로 재현한 것은 아니다.
