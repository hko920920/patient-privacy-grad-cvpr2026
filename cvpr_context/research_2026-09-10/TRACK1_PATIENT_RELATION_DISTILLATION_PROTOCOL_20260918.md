# 환자 관계 보존 증류: 첫 통합 개발 패키지 v1

작성일: 2026-09-18. 큰 단계2의 실행 설계·핵심 연산 구현이다. 관계 증류를 주력 가설로 유지하고 조건 대조 diffusion은 병행하지 않는다. 방법 우열이나 성능이 확인된 상태는 아니다.

이전 [설계 비교](TRACK1_PATIENT_RELATION_DISTILLATION_COMPARISON_20260918.md)의 A/B/C/D를 구체적인 설정으로 옮겼다. 이 문서와 JSON은 **첫 고정 recipe**이며, 전체 runner의 실행 연결은 아직 검증 전이다. 환자 이미지·모델·GPU 실행 없이 기존 metadata, 공개 checkpoint 파일 hash, CPU 배열만 사용해 준비했다.

## 1. 검증할 주장과 출력

> 환자 내부 class-centroid 대응 정보를 환자-DP로 보호하고 영상·대응관계로 전달하면, 같은 점별 정보만 증류하는 것보다 다른 시각 학습자에게 유용한 자료를 만들 수 있는가?

출력은128장과 점별/관계 역할,32개 pair의 인덱스, 고정 학습 규칙이다. 실제 환자 ID는 출력하지 않는다. 합성 label은 최적화 목적의 label이며 전문가가 확인한 영상 소견이 아니다. 평균·2차 모멘트 정합이나 Gaussian DP를 새 원리라고 주장하지 않는다. 표준 delta-feature moment matching이 C의 관계 연산과 동일한 부분은 그대로 인정한다.

## 2. 자료와 독립성

| 역할 | 환자 / 영상 | 기흉 양성 환자 | mixed 환자 |
|---|---:|---:|---:|
| P 기존 공개 실자료 | 672 / 813 | 6 | 3 |
| Q 과거 classifier-selection, private 출처 | 2,027 / 5,097 | 114 | 102 |
| V 기존 method-development | 2,026 / 5,047 | 114 | 104 |

Q는 원래 private_train이며 이미 calibration과 Rwide에 소비됐다. 공개자료나 미사용 validation으로 바꾸지 않는다. P/Q/V의 환자·영상·기록된 SHA 분리는 기존 명부에서 다시 확인한다. 원 역할표는 수정하지 않는다. Expert532와 reserved4213의 명부·pixel·prediction은 이번 실행 입력이 아니다. 이전 Q80 head/LoRA 실패는 Q2027의 비교 셀을 대신하지 않는다.

V도 여러 선행 실험에서 소비한 개발자료다. 이 패키지는 새로운 독립 확인이나 final 평가가 아니다. Private 정보 접근은 통계 추출 단계에 한정하고, 합성 최적화의 재사용 입력은 보호 전/후 요약으로 분리한다. 비DP 개발 기록은 소급해 DP가 되지 않는다.

## 3. 고정 encoder와 특징

| 용도 | 설정 |
|---|---|
| 증류 E1 | 공개 일반 DINOv2 ViT-B/14,768차원 CLS, register 없음 |
| 차원 축소 | P813만의 환자 균등 가중 PCA16, 라벨 미사용, whitening 없음 |
| 전이 E2 | 공개 ImageNet ResNet18,512차원 pooled feature, 분류기 fc 제거 |
| 입력 | grayscale→RGB, 종횡비 유지224 letterbox, ImageNet normalization |
| 모델 상태 | eval, 모든 parameter/buffer 고정, FP32, TF32 off |
| 특징 경계 | raw L2 normalize; E1은 공개 평균 중심화/PCA 후 norm floor1e-6로 L2 normalize |

E1의 PCA는 각 public 환자의 이미지 가중치 합을1/672로 두어 구한다. 고유값 내림차순, 각 축의 최대 절대 loading을 양으로 고정한다. 실제 projection을 계산해 hash를 저장한 뒤 Q 특징을 추출한다. 성능을 보고 차원을 바꾸지 않는다.

DINO 공식 checkpoint는 기존 local SHA `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`로 결속한다. ResNet18은 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`다. DINO의224 입력은 local timm의 dynamic-resolution 경로로 구현하고 원 weight load·positional interpolation·input gradient를 runtime에서 확인한다. 학습되지 않은 새 weight를 사용할 수 없다.

[DINOv2 모델 카드](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)는 LVD-142M 사전학습을 명시한다. 이 checkpoint의 NIH 환자별 중복 부재가 증명됐다고 부르지 않는다. [RAD-DINO 모델 카드](https://huggingface.co/microsoft/rad-dino)는 NIH-CXR 학습을 포함하므로 이번 E1로 쓰지 않는다.

E2의 architecture는 과거 분기에도 쓰였다. 여기서 제외하는 것은 **이번 증류의 gradient, bank/checkpoint/β 선택**이다. 프로젝트 전체에서 처음 보는 모델이라는 주장은 하지 않는다. E2 성능을 본 뒤 bank를 수정하면 이후 E2는 선택 이력이 있는 모델로 기록한다.

## 4. 네 조건과 대응 초기화

| Arm | 목표 자료 | 점별 이미지 | 관계 이미지 | primary readout |
|---|---|---:|---:|---|
| A | P의 점별 통계 |128|0|점별 위험|
| B | P+Q의 점별 통계 |128|0|점별 위험|
| C | P+Q 점별 + 실제 mixed 대응 |64|32쌍=64|점별+관계 위험|
| D | C와 같고 Q102의 음성 centroid 대응만 교란 |64|32쌍=64|점별+관계 위험|

Bank seed101/211/307을 모든 arm에 대응시켜 처음부터12개 bank를 만든다. D의 private derangement는 사전 seed202609181로 한 번 고정해 세 bank에 공통 사용한다. 공개 mixed3의 대응은 바꾸지 않는다. C/D의 점별 통계와 평균 delta는 같고 관계 2차 모멘트만 달라야 한다. D는 비DP 대조이며 patient-local 감도를 그대로 붙이지 않는다.

총128개의 인덱스0:64는 음성,64:128은 양성이다. C/D의 점별 subset은0:32와64:96, 관계 pair는32:64↔96:128이다. A/B에는128장 전부를 점별 정보에 사용할 수 있게 한다.

초기화는 각 seed에서 public 환자64명을 hash 순으로 고르고 환자별 영상 한 장을 label과 무관하게 정한다. 영상 j를 cell j와 j+64의 공통 기반으로 사용한다. 모든 arm에서 초기128장과 RNG가 같다. 영상별 작은 독립 jitter(std .01)를 둬 pair 두 영상이 정확히 같아 관계2차 손실의 gradient가0에서 시작하는 문제를 피한다. 결과에 따른 이미지 선별은 없다.

## 5. 합성 최적화와 실제 학습 목적

이미지는28/56/112 grayscale residual pyramid를224로 bilinear 확대해1/√3로 합치고 공개 초기영상에 더한 뒤[0,1]로 clamp한다. 초기 jitter도 같은 표현에서 생성한다. 데이터 증류 입력과 평가 입력의 전처리는 같다. 이번 위험 보존 명세에는 augmentation을 넣지 않는다. 합성 쪽에만 augmentation을 넣고 같은 실자료 위험이라고 주장하지 않기 위한 고정 선택이다.

목표는

L_point = 1/2 Σ_c (||m_Sc−m_Tc||² + ||A_Sc−A_Tc||_F²)

L_relation = ||delta_S−delta_T||² + ||C_S−C_T||_F²

L_total = L_point + beta L_relation + .001 TV + .001 anchor.

Beta=1(C/D),0(A/B), Adam lr=.01, betas(.9,.999),eps1e-8,weight_decay0,500step, 마지막 checkpoint만 사용한다. TV는 수평·수직 인접 픽셀 절대차 평균의 합, anchor는 공통 jitter 초기영상과의 픽셀 제곱차 평균이다. 이 값들은 결과 기반 최적값이 아니라 첫 비교의 고정값이다.

Microbatch4로 처리하되 전체128장 특징에서 한 번 계산한 손실의 dL/dz를 각 encoder 재계산에 전달한다. Microbatch마다 모멘트 손실을 계산해 평균하지 않는다. 이 두-pass 구현은 한-pass full-bank autograd와 일치해야 하며, 추가 forward 비용도 총비용에 포함한다. CPU 배열에서는 이 gradient 연결을 검산했다. 실제 DINO·pixel에서의 대응은 아직 확인 전이다.

최종 산출물은 uint8 grayscale PNG다. Primary E1/E2 readout 모두 다시 decode한 PNG에서 특징·위험을 계산한다. 연속 최적화 값과 최종 PNG의 모멘트 차이는 quantization 실현 오차로 보고한다.

분류기는 기존 설계의 이차 위험을 ridge λ=.01,bias 없음으로 푼다. Primary는 합성자료만 사용하며 Q 원자료를 다시 읽지 않는다. C/D의 관계-loss-off 대조는 같은64장 point bank만 유지하고 beta=0으로 둔다. 관계 bank를 point bank에 몰래 추가하지 않는다. C−B는 자료와 학습 규칙의 전체 효과이며 C−D,loss-off와 의미가 다르다.

## 6. 평가와 판정

12개 bank를 모두 고정·hash한 뒤 V 예측을 계산한다. E1/E2 실자료 point/relation readout도 같은 패키지에서 진단 비교로 계산한다. 이 실자료 readout은 Rwide의0.679를 다른 모델의 상한으로 복사하는 대신 새 표현의 기준을 제공한다. Q에 직접 접근한 비DP readout을 보호된 산출물로 표시하지 않는다.

- Primary: E2 관계 활용 학습의 C−B,C−D. B−A와 C−A는 사적 정보 및 공개 기준 대비 맥락으로 함께 보고한다.
- 지표: bank별 AUROC/AP를 구한 뒤3개 bank 평균. Prediction ensemble 성능으로 바꾸지 않는다.
- 불확실성: 기존2000개 환자 cluster bootstrap draw를 모든 arm/bank에 공유하고95% 구간을 보고한다. 이 구간은 고정된3bank/encoder에 조건부인 개발 분석이다.
- 후속 DP 검토 신호: C−B와C−D가 각각 평균 AUROC≥.01, 양의 bank≥2/3, 평균AP 비감소를 만족. 임상/통계 유의성 또는 final 성공 기준은 아니다.
- C가 B보다 나아도 A보다 못하면 public-only 대비 실용적인 사적 추가 효용을 확보했다고 쓰지 않는다. 같은 결과표에서 드러내며 사후로 A를 숨기지 않는다.

기존 ResNet18 BCE는 secondary 호환성 평가다. 공개real16+synthetic16,batch32,400step,lr1e-4,seed11/23/37의 corrected kernel을 사용한다.4arm×3bank×3classifier seed=36run이며128장 모두를 요청 label로 사용하고 pair는 무시한다. 여기에서 실패했다고 E2 관계 활용 전이까지 자동 무효 처리하지 않는다.

Bank 반복, classifier 반복, 환자 bootstrap은 서로 다른 변동이다. 한 숫자로 합쳐 표본 수를 부풀리지 않는다. 유리한 한 bank만 남기거나 추가 bank를 결과마다 한 개씩 붙이지 않는다.

## 7. DP는 같은 설계의 후속 단계이며 자동 실행하지 않는다

기본 ε8,δ1e-5,add/remove-one-patient를 후보로 유지한다. Private Q의 한 환자 기여를 묶고 P의 정확한 통계를 나중에 더한다. d16에서 B는306좌표/감도√6, C는459좌표/감도3이다. B에 관계 비용을 강제로 부과하지 않는다. 각 mechanism의 Gaussian scale은 analytic calibration으로 계산한다.

Joint query는 class 존재 count와(C에서)mixed count를 포함한다. Exact public+noisy private 합계를 먼저 더한 뒤 정규화한다. Noisy class count는1로 floor, mixed count는[1,min(class counts)]로 제한한다. 이 규칙은 후처리이며 true count를 공개하는 것이 아니다.

DP에서 사용할 target 보정은 평균 unit ball 제한, covariance PSD화, trace cap으로 사전 선택한다. 보정은 필수 정리나 이미지 실현 보장이 아니다. 동일 보정 목표를 직접 통계 분류기와 이미지 정합에 사용하고 보정·실현 오차를 분리한다. 실제 synthetic H는 target 보정 여부와 무관하게 PSD다.

이번 구현은 query와 calibration 계산까지만 갖췄다. 실제 privacy-noise release·보호된 난수·composition ledger는 아직 구현/검증 전이다. 공개 고정 seed의 잡음을 실제 DP release로 쓰지 않는다. 여러 요약/예산/독립 noisy seed 산출물을 함께 공개할 때의 composition을 별도로 계산한다. 하나의 이미 보호된 요약으로 여러 bank를 만드는 것은 후처리다. 비DP 원통계·성능 기반 선택·환자별 cache·noise seed를 공개하지 않는다.

## 8. 실행 단계와 비용

다음 runtime 구현은 하나의 패키지다: 입력 결속→public-only profile→E1/PCA/통계→12bank 최적화와 seal→E1/E2 평가→BCE36run→산출물 검산. 기존 legacy classifier import는 계속 차단한다. 완료 파일 overwrite를 금지하고 source/optimizer/RNG/hash가 같은 중단 재개만 허용한다.

첫 profile은 공개 초기 bank만 사용해 최대4update/20분으로 제한한다. 원본 pixel 연결, checkpoint strict-load, DINO224 input gradient, 전체bank/두-pass 동일성을 확인하고 시간·peak VRAM·IO를 기록한다. Private 통계와 개발 점수는 이 profile의 선택 기준이 아니다.

Profile에서 바꿀 수 있는 것은 동일 목적을 보존하는 microbatch/IO 경로뿐이다. 전체 bank의 gradient가 같아야 한다. 목적·encoder·500step·arm을 바꾸는 경우 기존 고정 recipe의 새 버전으로 기록한다. 일부 개발 점수를 본 뒤 뒤 arm을 바꾸지 않는다.

본 연산량은12bank×500step×128영상이며 각 step의 두 encoder pass 및 backward를 계산해야 한다. 여기에 P+Q5910장/V5047장의 두 encoder 추론과36개 BCE학습이 있다. 실제 profile 전에는 완료 시간을 확정하지 않는다. 현재 준비 작업의 metadata·hash·CPU 검산 시간과 실제 GPU 비용을 혼동하지 않는다.

## 9. 현재 구현과 미완료 항목

코드 위치: `code_working/relation_distillation/`.

구현: 환자별 모멘트와 bounded query, public/private 합계의 정규화, private-only derangement, optional moment repair, 직접 ridge readout, Gaussian scale, 전체bank differentiable loss/feature gradient, 설정 JSON과 기존 명부/공개 weight 결속.

검산: raw image-feature 배열의 환자균등 위험과 이차식, 반복 방문의 환자 질량, svec isometry, 감도3/√6, pairing의 평균 불변/2차 변화, joint-moment와delta 연산 동일성, 불가능한noisy목표, Gaussian calibration, 두-pass gradient를 CPU에서 검사한다. 이것은 실제 영상 연결 또는 효용 관문이 아니다.

남은 구현: image decoder/cache, 공개 PCA와 DINO 실제 경로, pyramid optimizer, E1/E2/PNG readout orchestration, BCE 공급자, resume/산출물 verifier, 향후 실제 DP release. `runtime_ready=false`, `final_ready=false`를 유지한다. 현재 full64·LoRA 실패, Rwide 결과와 잠긴 자료는 변경하지 않는다.
