# Receiver 조건별 signal 첫 구현 결과 — 2026-09-22

판정: **새 조건별 gradient의 첫 기술 연결 검사가 통과했다.** CPU 9개와 실제 BioViL 공개 4장·microbatch1/4 검사를 완료했다. A1/A2 성능 비교나 새로운 논문 기여가 성립했다는 결과는 아니다.

사용자의 “기록하고 하나씩 실험” 요청에 따라 [개발 방향과 이번 범위](RECEIVER_DISTILLATION_DEVELOPMENT_PLAN_20260922.md)를 먼저 기록하고 실행했다. 현재 관계 PRRD-C와 augmean 확대는 보류한다. 과거 결과·계약은 변경하지 않았다.

## 구현한 것

기존 PRRD 파일 33개는 hash가 모두 유지됐다. 별도 `code_working/receiver_distillation`에 다음을 추가했다.

| 파일 | 역할 |
|---|---|
| signals.py | 공개 K4 조건·고정 head, 영상별 CE gradient, 환자/class 집계, 조건·encoder 평균 cosine loss |
| objectives.py | 전역 signal one-pass 및 동일 분할 two-pass replay, 기존 anchor/TV를 전체 목적에서 한 번만 적용 |
| test_signals.py | 독립 CE/autograd 및 환자 가중 집계·조건 재현·두 encoder CPU 검사 |
| verify_public.py | 실제 공개 이미지·checkpoint·P-only projection·코드·허용오차 사전 결속과 gradient 검사 |

BioViL의 기존 정규화·P-only PCA16·q95 제한을 재사용했다. Raw 관계·PCA32·whitening은 추가하지 않았다. Fixed head는 공개 seed와 encoder ID로 생성하므로 두 번째 encoder를 추가해도 기존 BioViL head가 바뀌지 않는다.

K4는 서로 다른 고정 augmentation/head 조건 4개를 **별도로** 보존한다. 같은 조건의 상대 affine/brightness를 real native 영상과 synthetic224 영상에 적용한 뒤 기존 preprocess를 사용한다. 과거 이미지별 K4 평균 target과 다르며 기존 cache는 재사용하지 않았다.

각 class의 환자 내부 방문 평균 → class-present 환자 평균 → class별 1/2 가중 합으로 CE gradient target을 계산한다. P/Q는 환자별 gradient의 class 합과 class-present 환자 수를 더한 뒤 나눈다. 환자별 영상 수가 많은 것을 더 큰 환자 가중치로 바꾸지 않았다.

새 목적은 조건별 고정 CE gradient의 cosine 정합이다. 과거 B의 moment·기능·관계항을 유지한 한 항 변경으로 부르지 않는다. 기존 B와의 비교는 전체 package 비교이며 pairing만의 인과적 효과는 아니다. 첫 A2는 equal-weight encoder mean이다. Smooth-max, CVaR, gradient balancing 또는 DP-aware 신호 압축을 구현했다고 주장하지 않는다.

## 검산

| 검사 | 결과 |
|---|---:|
| CPU 독립 검사 | 9개 PASS |
| 영상별 CE 해석 gradient 대 autograd 최대 오차 | 1.11e-16 |
| 위 gradient의 feature 미분 최대 오차 | 2.22e-16 |
| 환자/class 가중 CE 목적의 독립 autograd 오차 | 6.94e-17 |
| 두 encoder CPU one/two-pass 최대 gradient 오차 | 4.66e-10 |
| 공개 저장 특징으로 재계산한 실제 target 최대 오차 | 5.55e-17 |
| 실제 BioViL one/two-pass | microbatch1/4 모두 PASS |
| 실제 최대 parameter gradient 오차 | 1.49e-08 |
| 실제 최대 parameter gradient 상대 L2 오차 | 5.21e-08 |
| 실제 encoder-input pixel gradient 차이 | 0 |
| 실제 loss 차이 | 0 |
| 조건 signal 단독 이미지 gradient norm | 9.591992, finite |
| Encoder 가중치/buffer·고정 target·renderer parameter | 불변 |

기존 loss 허용오차2e-6, gradient 허용오차2e-7+reference peak×5e-4를 유지했다. 두 경로는 renderer부터 같은 microbatch 분할을 사용한다. 서로 다른 microbatch의 bitwise 동일성을 주장하지 않는다.

CPU 검사는 mixed 환자, 빈 class, 방문 반복, P/Q 결합·중복 환자 거부, 조건/head RNG 재현, target detach, A2의 공통 BioViL target/loss 유지, feature finite difference를 포함한다. 여기서 사용한 두 번째 encoder는 구성 배열용 작은 함수이며 DINO 실검증이 아니다.

실제 검사는 P의 서로 다른4명에서 음성2/양성2를 사전 hash 규칙으로 정했다. 저장된 K×영상×feature에서 독립 CE/autograd로 target을 다시 계산했다. 별도의 signal-only probe로 prior/TV 없이도 실제 이미지 gradient가 생기는 것을 확인했다. 모든 검사는 첫 시도에서 통과했으며 허용오차 변경은 없었다.

## 실제 실행량과 비용

- 공개 원영상4장만 읽음. K4 target 추출16 image-forward.
- 전체 실제 BioViL forward 116 image-evaluations, backward 68 image-evaluations.
- Optimizer update0, 정식 합성 bank0, Q/V 픽셀0, 수신 모델0, DINO0, DP/Expert/Reserved0.
- CPU 검사 0.80초, 실제 모델 검사 프로그램 11.45초.
- Peak CUDA allocated 5554.69MiB, reserved 5914.00MiB.
- 최초 작업 예상20–30분; 구현·기록 포함 이 보고서 작성 시점 약 14.6분. GPU 프로그램 시간과 전체 작업 시간을 구분한다.

이 peak에는 비교 기준인 graph-retained one-pass가 포함된다. 정식128장 two-pass의 비용 측정치가 아니며, 기존 B의49분이나 이번4장 시간을 새500회 본학습 비용으로 확정하지 않는다. 측정용 PNG나 bank는 만들지 않았고 변환 원픽셀도 저장하지 않았다.

## 다음 한 단계

A1용 고정 조건 target 준비와 실행 연결을 구체화한다. 환자/class 규칙·조건/head·P-only 변환은 이번 코드와 동일하게 결속한다. A1/A2 본 효용 비교 전에 두 번째 encoder의 공개 변환, 공통 optimizer/계수, 실제 비용 상한, AUROC 최소 개선·AP/source 허용 감소를 기록한다. 현재 0.598이나 보고서의 비용 추정을 자동 채택하지 않았다.

A1/A2는 feasibility 비교다. DenseNet/V는 개발용이고, 별도 receiver/Expert/Reserved는 계속 닫는다. 미래 DP는 private class counts·clipping·잡음·공동 공개 composition을 별도 계약으로 계산해야 한다. 이번 비DP signal을 보호된 release라고 부르지 않는다. 이 단계에서 연구의 큰 단계2 전체가 완료된 것도 아니다.

근거 파일: [CPU 검사](../../code_working/_reports/receiver_condition_step1_20260922_v1/cpu_attempt1.json), [사전 공개 계약](../../code_working/_reports/receiver_condition_step1_20260922_v1/public_attempt1/public_contract.json), [실제 모델 검산](../../code_working/_reports/receiver_condition_step1_20260922_v1/public_attempt1/verification.json). 검사한 신규 소스는 `implementation_snapshot`에 보존했다.
