# Receiver 재사용 증류: 개발 방향과 첫 구현 계약 — 2026-09-22

현재 관계 PRRD-C와 B_augmean의 추가 실행은 보류한다. 기존 음성 결과와 실행 계약은 보존한다. 새 방향은 source에서 잘 재현된 판별 정보가 다른 영상 모델에서도 쓰이도록 만드는 증류이며, 장기적인 기여 후보는 같은 환자-DP 예산에서 여러 encoder 신호의 clipping·잡음 부담을 제어하는 것이다. 그 연산은 아직 완성되지 않았다. 두 encoder, 환자 평균, 조건 저장, 단일 DP 요약만으로 신규성을 주장하지 않는다.

## 확정한 해석과 비교

- P=공개 672명/813장, Q=보호 대상 2,027명/5,097장, V=반복 사용한 개발자료. DenseNet은 개발 receiver이며 합성 loss에 제외됐다는 이유로 unseen이라 부르지 않는다.
- A1은 새 공통 조건별 선형-head gradient 경로의 BioViL-only 비교, A2는 같은 경로에 두 번째 공개 encoder를 더한 equal-weight mean 비교다. 첫 A2에는 smooth-max나 새 balancing을 동시에 넣지 않는다.
- BioViL target, head, augmentation, 환자·class 가중치와 공통 설정은 A1/A2에서 동일하다. Encoder 추가가 BioViL target 자체를 joint clipping으로 바꾸게 하지 않는다. 이번 비DP 경로는 그런 joint clipping을 사용하지 않는다.
- B→A1은 전체 목적이 달라지는 개발 비교다. 조건 pairing만의 독립 효과라고 해석하지 않는다. A2 개선도 두 번째 encoder 자체 효과와 diversity 효과를 아직 분리하지 못하므로, 후속 DINO-only 등은 근거가 생긴 뒤 판단한다.
- 새로운 K4는 네 개의 고정 augmentation/head tuple이다. 과거 이미지마다 네 변환을 평균한 target과 다르며 기존 augmean cache를 재명명하거나 재사용하지 않는다.
- LGM은 real/synthetic에 각각 증강하며 동일 RNG replay가 필수인 방법이 아니다. 조건 tuple 재사용의 근접 선행은 Dosser다. 이 구현은 어느 선행의 전체 재현도 아니다.
- 한 seed 결과는 개발 판단이고 환자 bootstrap은 해당 bank를 조건으로 한 불확실성이다. 부정적 결과는 해당 recipe의 추가 투자를 멈출 근거이지 전체 multi-model 연구의 불가능성 증명이 아니다.
- 최종 receiver/Expert/Reserved는 계속 닫는다. NonDP Q 기반 설계 선택을 향후 DP로 소급하지 않는다. 고정 메커니즘의 보장과 연구 산출물 전체의 보장을 구별한다.

## 이번 한 작업: 공개 조건별 signal → 실제 이미지 gradient 연결

예상 총 20–30분. 정식 bank 학습 전에 이번 새 경로만 검산한다. 기존 PRRD production 파일은 변경하지 않고 별도 receiver_distillation 패키지로 구현한다.

1. 기존 BioViL point 특징(P-only PCA16, 기존 norm/scale, whitening 추가 없음)을 재사용한다. 이는 첫 기술 검사와 A1 시작 경로의 기본 선택이며 PCA32가 검증됐다고 적지 않는다. 두 번째 encoder용 변환은 아직 미구현이다.
2. K=4, public seed101. 동일한 조건 k의 affine/brightness 파라미터를 모든 real/synthetic 영상에 적용한다. 기존 ±3도, ±1% 이동, scale/brightness 0.98–1.02 범위를 유지하며 native real / 224 synthetic의 공통 상대좌표 변환 후 기존 preprocess를 사용한다.
3. Encoder ID와 k로 고정한 2-class linear head W,b를 사용한다. PyTorch Linear의 기본 uniform 범위 ±1/sqrt(d)를 공개 난수로 생성한다. CE의 영상별 weight/bias gradient를 해석식으로 계산하고 독립 autograd와 대조한다.
4. 각 class에서 환자 내부 방문을 먼저 평균한 후 class-present 환자들을 동일 가중 평균하고, 두 class를 각 1/2로 합한다. P/Q 결합은 class별 환자 gradient 합과 class-present 환자 수를 더한 뒤 나눈다. 방문 수와 조건 수가 환자 질량을 늘리지 않는다. 빈 class의 항은 0, 남은 class를 임의로 재정규화하지 않는다.
5. 합성 측은 class당 이미지 동일 가중치의 CE gradient를 구한다. 각 조건의 target과 cosine loss를 계산하고 조건/encoder에 equal-weight mean을 쓴다. 기존 public template anchor/TV는 전체 bank에서 한 번만 더한다. 과거 moment/functional/relation 항을 이 새 경로에 몰래 섞지 않는다.
6. 공개 4장(각 class 2명)의 target을 새로 만들고 동일 renderer 분할의 one-pass/two-pass를 microbatch1/4에서 대조한다. Renderer step500은 모든 pyramid 수준을 켜는 **검사 위치**이며 optimizer update는 0이다.
7. 기존 gradient 수치 기준(loss abs≤2e-6; max gradient error≤2e-7+5e-4×reference peak)을 그대로 사용한다. 실제 parameter 및 encoder-input pixel gradient, 고정 encoder weights/buffers, target 불변을 확인한다.
8. CPU에서는 환자/방문 가중치, mixed/빈 class, P/Q 결합, CE gradient, 조건 replay, A2가 공통 BioViL block을 바꾸지 않는지, 작은 두 encoder replay를 검산한다.

이번 작업은 Q/V 픽셀, 전체 특징 추출, 정식 128장·500 update, DINO 로딩, 수신자 성능, DP, Expert/Reserved를 실행하지 않는다. 저장은 작은 공개 기술 JSON 및 코드만이며 새 학습 bank는 만들지 않는다.

## 다음 단계와 남은 실행값

기술 검사가 통과하면 같은 경로의 실제 target 준비와 A1/A2 실행 연결로 간다. 정식 효용 실행 전 common recipe, 두 번째 encoder/P-only projection, 실제 비용, AUROC 최소 개선·AP·source 허용 감소를 수치로 동결한다. 0.598은 자동 채택하지 않는다. 128장·500회·seed101은 기존 비교 규모를 유지하는 시작 후보이며 새 경로의 속도를 기존 B 49분으로 확정하지 않는다. 실패 후 K/LR/encoder/loss/seed 자동 탐색과 자동 fallback은 없다.

## 근접 선행과 기여 경계

[LGM](https://arxiv.org/abs/2511.16674), [HMDC](https://arxiv.org/abs/2409.14538), [model-pool](https://arxiv.org/abs/2402.13007), [Dosser](https://arxiv.org/abs/2508.01749), [DP-NTK](https://arxiv.org/abs/2303.01687), [DP-MEPF](https://arxiv.org/abs/2205.12900), [DP-KIP](https://arxiv.org/abs/2301.13389), [DP-GenG](https://arxiv.org/abs/2511.09876)를 기준으로 본다. 이 문서는 이전 원문·코드 검토를 기록한 것이며, 이번 작업에서 문헌 전체를 다시 검색했다는 뜻은 아니다.

향후 DP에서는 class sums/counts, receiver/condition blocks와 최종 환자 contribution의 감도를 함께 명시해야 한다. 비DP의 실제 class 분모를 공개 상수로 간주하지 않는다. 고정 public N, replace-one, 최종 norm C 조건의 2C/N과 기존 add/remove 정의를 혼동하지 않는다. 같은 보호 요약의 여러 초기화는 후처리이며 별개 요약 공동 공개는 composition 대상이다. 차원×잡음분산 증가만으로 downstream 효용 저하나 새 방법의 우위를 증명하지 않는다.

