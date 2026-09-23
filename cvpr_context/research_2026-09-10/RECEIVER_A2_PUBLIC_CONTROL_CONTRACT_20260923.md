# A2 공개전용 대조 — 2026-09-23

큰 단계2의 개발 대조다. 사용자 요청에 따라 P-only A2 한 bank의 준비·학습·저장·개발 평가를 연속 수행한다. 기존 seed101 P+Q A2를 재사용하여 Q를 추가한 효용을 비교한다.

- BioViL+DINO equal mean, K4 조건/head seed101, 각 encoder P-only PCA16/q95를 유지한다.
- 128장·200회·seed101·microbatch16, AdamW lr0.01와 기존 해상도 일정1/33/65/97/129/161, anchor0.01/TV0.0001을 유지한다.
- 두 encoder target만 P+Q에서 P로 바꾼다. 원본 저장파일의 P_sums/P_counts/P_gradient를 분리해 그대로 저장하고, 공개 환자 명부의 class-present counts와 독립 나눗셈을 1e-12로 검산한다. 학습용 새 파일에는 Q/pooled block을 포함하지 않는다.
- 환자/class 내부 영상 평균 → 해당 class가 있는 환자 평균 → class별0.5 가중을 유지한다. 양성 환자가 적다는 이유로 Q 분모를 섞거나 새 scale을 고르지 않는다.
- 기존 A2 seed101의 공개 template와 정확히 같은 초기 renderer·optimizer·RNG에서 새로 시작한다. 기존 학습된 checkpoint를 이어 쓰지 않는다. 보존된 step0은 비교용 증빙이며 실제 새 학습의 입력 checkpoint가 아니다.
- 검증된 repeat_runtime의 학습/저장/재개/PNG 루프를 직접 재사용한다. 새 adapter는 P-only job 검증과 기존 시간 예산 처리를 연결한다. 변경되지 않은 gradient/실행기 검증을 반복하지 않는다.
- 최종200 PNG를 고정하고 나서 기존 V 특징과 2,000 paired 환자 bootstrap draw를 재사용한다. DenseNet AUROC/AP 및 BioViL에서 기존 A2(P+Q)−새 A2(P)를 보고한다. 중간 성능·checkpoint 선택은 없다.
- 한 seed의 고정 bank 비교다. CI는 V 환자 표집 불확실성이며 합성 seed 불확실성을 포함하지 않는다. DenseNet/V는 개발 자원, BioViL은 제작 source다.
- 예상 준비·검사10~20분, 합성·저장약75분. 합성 worker 상한은 기존 A2와 같은7,200초이며 재개해도 누적한다. 평가·기록은 별도다.
- 각 bank의 최초·마지막두/최종 checkpoint와 모든 receipt/trace를 보존하고, 이번 bank 내부의 중복 payload만 경로·해시 확인 후 정리한다. 기존 결과·캐시를 삭제하지 않는다.
- 추가 seed·500회 연장·새 목적·DP·Expert·Reserved·확인용 receiver로 자동 확대하지 않는다. 새 P/Q 특징 추출은 없다.

정확한 목표·코드·초기상태·평가 의존성은 code_working/_reports/receiver_a2_public_s200_20260923_v1/의 준비/실행 계약과 해시에 결속한다. 본 문서는 실행 범위이며 효용 결과는 아니다.
