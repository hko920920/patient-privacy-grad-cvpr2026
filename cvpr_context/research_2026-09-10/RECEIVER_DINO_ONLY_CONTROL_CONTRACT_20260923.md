# DINOv2 단독 대조 — 2026-09-23

현재 큰 연구 단계는 2의 개발 검증이다. 완료된 A1/A2의 양성 결과를 유지하면서, DINO source 선택과 두 source 결합의 역할을 구분한다. 사용자 최신 지시에 따른 한 bank 작업이며, 결과에 따라 자동 확대하지 않는다.

- 실행: dev_DINO_condk4_101_s200, 128장, 200 update, seed101, microbatch16.
- A2의 DINO checkpoint·P-only PCA16/q95·K4 head/affine/brightness·P/Q 환자/class 가중 목표를 그대로 재사용한다. 새 목표 추출 없음.
- 손실: DINO K4 gradient-cosine 평균을 단일-source 가중치 1로 사용하고 anchor 0.01 / TV 0.0001을 한 번 적용한다. BioViL 손실은 제외한다.
- 공개 template, 64/64 label, AdamW lr0.01 및 기존 나머지 설정, pyramid 활성 1/33/65/97/129/161을 유지한다.
- 합성 worker 상한 60분(시작·저장·PNG 포함, 중단/재개 누적). 사전 예상 합성 35–45분, 연결·검사·평가·기록 포함 50–75분. 준비·평가 시간은 별도 기록한다.
- 최종 200회 PNG를 고정한 뒤 기존 BioViL/DenseNet V 특징과 동일한 2,000회 환자 bootstrap을 사용한다. 학습 중 효용 평가나 checkpoint 선택 없음.
- A1/A2의 bank·예측·bootstrap·비용은 재사용한다. A2−DINO와 DINO−A1의 AUROC/AP, BioViL 유지, 실제 비용을 함께 보고한다.
- 새 수치 통과선을 결과 이후 만들지 않는다. 유의하지 않은 차이는 동등성 입증으로 해석하지 않는다. 단일 seed·개발 V/receiver의 조건부 결과다.
- 추가 seed, 500회 연장, DP, Expert, Reserved, final receiver는 범위에 없다.

기존 실행기·저장/재개·PNG 및 DINO 목표 준비 검사를 재사용한다. 변경된 source 선택/단독 가중치에 대해 공개 P4, K4, microbatch4 한 조건의 실제 retained-graph / two-pass gradient만 대조한다. 초기 renderer·optimizer·난수·활성 상태는 A2 step0과 정확히 비교한다.

착수 전 과거 구현 전체 해시 검사에서 test_target_runtime.py 한 파일이 달랐다. 기존 snapshot과 비교하니 내부 handoff가 없을 때 테스트를 건너뛰는 2줄만 추가되어 있었다. 실제 학습/평가 코드 해시는 모두 같았다. 사용자 변경을 보존하고 정확한 diff와 양쪽 해시를 campaign_contract.json에 기록했다. 첫 중단은 GPU/산출물 생성 전이었다.

기계 판독 계약·실행 결속·로그·검사·최종 결과 경로:
code_working/_reports/receiver_dino_only_s200_20260923_v1/

합성자료와 목표는 NONDP_INTERNAL_Q_DERIVED 내부 자료다. 환자-DP가 적용됐다고 주장하지 않는다.

