# 기존 DP PNG의 ResNet18 추가 개발 평가 계약

2026-09-24 KST. 연구 단계2의 제한된 추가 수신자 평가다. 사용자 `실행` 지시로 아래 범위만 수행한다. 예상20~40분, 준비 시작2026-09-23 17:27:50 UTC부터 전체45분 상한(18:12:50 UTC). 정상 연결·특징 추출·readout·평가·기록은 중간 승인 없이 이어간다.

## 고정 질문과 산출물

기존 A2 DP1/DP2와 DINO DP1/DP2를 동일한 추가 개발 수신자가 읽을 때 어느 효용과 비용의 절충이 관측되는가? 기존 A2 공개전용 seed101 bank와 공개 P 실영상 readout을 함께 비교한다. 네 보호 bank와 공개전용1bank는 모두128장·200회·합성 seed101 최종 PNG이며 재생성하지 않는다. DenseNet 주 개발 결과와 BioViL 결과는 보존·병기한다.

ResNet18은 이전 연구 개발에서 이미 사용한 architecture다. 이번 평가는 추가 개발 비교이며 최종 unseen receiver 검증이 아니다. Expert·Reserved·확인용 수신자 접근은 없다.

## 결과 전에 고정하는 평가 연산

- 체크포인트: torchvision ImageNet ResNet18 IMAGENET1K_V1, `resnet18-f37072fd.pth`, SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. 전체 state strict load 후 fc=Identity, eval, 모든 parameter 동결. 기존 의료 fine-tuned checkpoint를 선택하지 않는다.
- 모든 실제/PNG 영상은 같은 CPU tensor 전처리를 쓴다. grayscale uint8/255, FP32 antialias bilinear short256, center224, RGB 복제, ImageNet mean/std. 기존 `prrd_v3.encoders.preprocess(...,'recipient')`를 재사용한다.
- 평균 풀링512차원에서 추가 raw L2 없이, 공개 P813장/672환자만으로 환자 균등·방문 균등 비지도 PCA128과 projected norm의 환자 가중 q95를 한 번 정한다. whitening 없음. `z=a/max(q95,||a||)`. 공개 fit은 FP64, 실제 공통 projection은 저장계수를 FP32로 변환한다. DenseNet의 현행 P-only PCA128/q95 원칙을 유지한다. 과거 v1의 full512/L2 ResNet 수신 규칙과 동일한 구현이라고 하지 않는다.
- 점별 ridge readout: ridge0.1, bias 없음, class labels−1/+1, class 각각0.5. PNG는 class별64장 균등. P 실영상은 class별 환자 균등→환자/class 방문 균등. 관계항 beta0. 기존 `learn`/`bank_moments`를 재사용하며 직접 가중 normal equation과 대조한다.
- P 기반 projection과6개 readout은 V 특징 추출·성능 계산 전에 동결한다. P 실영상 기준은 동일 projection에서 계산하므로 추가 모델 추출을 요구하지 않는다. 어떤 readout도 V 결과로 계수·차원·표현을 바꾸지 않는다.
- P813+V5047+5bank×128=6500장의 새 forward를 실행한다. 변경된 feature 위치 확인은 P 첫16장의 수동 avgpool 경로16회 forward만 추가한다. FP32·microbatch16·TF32 off. backward·optimizer·새 합성 bank·새 Q DP query는 모두0.

## 검산과 결과 해석

기존 최종 PNG/label/hash 검사를 재사용하고, 고정 encoder parameter/buffer 불변, P만 사용한 PCA/scale, feature·명부 순서, 직접 patient/class 가중 readout을 확인한다. 수동 avgpool 경로 exact, FP64 projection 대조≤1e-5, centre/orthogonality≤1e-10, readout solve≤1e-9, 독립 sklearn bootstrap 대조≤1e-12를 결과 전에 고정한다. 허용오차를 통과할 때까지 바꾸지 않는다.

기존 V2026환자/5047장과2000회 patient-cluster bootstrap draw를 재사용한다. AUROC/AP 및 각 bank의 구간, 네 A2−DINO 교차 비교를 모두 보고한다. A2 DP−A2 공개전용, 각 보호 bank−공개 실영상 기준, 각 방법 noise2−noise1도 함께 기록한다. 잡음 번호끼리 대응된 난수라고 해석하지 않는다. 네 교차 비교는 같은 네 bank를 공유하므로 독립 반복4회가 아니다.

두 잡음의 성능 산술평균은 기술통계다. 환자 구간은 해당 고정 bank/실현 잡음에 조건부이고, 합성·잡음 모집단 분산이나 등가성을 증명하지 않는다. 새 개발 수신자 결과 하나로 일반 우위·안정성·논문 기여를 확정하지 않는다. DenseNet의 불확실한 우위를 숨기거나 주평가에서 제외하지 않는다.

수신자별 이득이 다르면 사용 조건별 절충으로 보고한다. 추가 가치가 없으면 현재 A2의 비용을 정당화할 근거가 부족하다는 결과를 보존한다. 좋은 결과가 나올 때까지 다른 수신자·표현·계수를 자동 탐색하지 않는다.

## 개인정보와 종료 경계

기존 DP PNG와 공개 P로 학습한 readout의 재사용이다. Q 픽셀/환자별 특징/새 보호 query·새 잡음·합성학습은 없다. 기존 네 요약의 공동 공개 기본 합성 상한(32,4×10⁻⁵)은 증가하지 않는다. 이 내부 V 평가 보고서를 과거 비DP 개발 전체까지 보호한 산출물이라고 부르지 않는다. 외부 업로드·공개는 하지 않는다.

코드·모델·P/V 명부·최종 bank·기존 평가와 privacy ledger 해시를 실행 전에 JSON 계약에 결속한다. 시간 상한, 비유한값, 자료·코드 binding 실패에서 멈추고 흔적을 남긴다. 이 한 수신자의 정해진 비교와 기록을 완료하면 종료한다. 추가 bank·noise·seed·예산·학습 연장·Expert·Reserved·확인용 수신자로 자동 확대하지 않는다.
