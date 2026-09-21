# PRRD A/B/C/D seed101 단계적 개발 계약 — 2026-09-21

사용자의 최신 요청으로 첫 실행 범위를 A/B/C/D 각1개, seed101의 **4bank**로 바꾼다. 이 변경은 Q 특징 추출과 새로운 V 성능을 보기 전에 기록한다. 원래30bank 계획은 이력으로 보존하고 남은26bank는 자동 실행하지 않는다.

원계약 PRRD_FIRST_NONDP_EXECUTION_CONTRACT_20260918.md의 방법·환자 가중치·특징·계수·학습량은 그대로다. 실행 분할은 최근 공개 검사에 따른 microbatch16이다. 원계약의 “30bank 전체 완료 후 평가” 조건만 이번 개발 목적에 맞게 “이번4bank 전체 완료·고정 후 동시 평가”로 대체한다.

| 조건 | 사용 정보 | 출력·학습 |
|---|---|---|
| A | 공개 P 점별 통계 |128 marginal, beta0|
| B | P+Q 점별 통계 |128 marginal, beta0|
| C | B의 점별＋실제 환자 raw 관계 |64 marginal＋32pairs, beta1|
| D | B의 점별＋Q 내부 대응 교란 raw 관계 |C와 동일, 공개pair3명 고정|

모두 seed101, 128장, 500 successful updates, FP32 encoder/renderer·FP64 작은 통계, AdamW lr.01/wd0, ridge.1, kappa.1, eta1이다. 관계 제한은 rho²=.1×w0ᵀMw0이며 A/B에는 관계항이 없다. 기존 pyramid/증강/통계·기능·영상 loss와 마지막500-step 채택 규칙을 유지한다. Microbatch는 checkpoint run_spec와 signature에 들어가므로 재개 중 바꿀 수 없다.

실제 첫 실행 IDs는 main_A_101, main_B_101, main_C_101, main_D_101이다. Public profile parameter, 중단된 과거 budget diagnostic, 기술 검사 bank는 재사용하지 않는다. 네 bank는 새로운 parameter와 독립 optimizer에서 시작하되 같은 공개 template·seed 규칙을 따른다.

실행 묶음은 Q 준비 → 환자별 목표 통계 생성·독립 검산 → 실제 파일 hash 결속 → 네 bank500회·PNG 저장 → 전체 봉인 → 지정 수신자의 V 평가다. 정상 구간에서는 단계나 bank마다 별도 승인을 요구하지 않는다. 데이터/수치 검증 실패, 안전한 재개 불가, 작업 상한, 계약 범위 변경이 필요할 때만 중단하고 보존한다.

실측 환산 합성은 약3시간20분이다. Q 준비·검산·저장·평가를 포함한 보수적 운영 상한은 assistant가 착수 전에 알린 **6시간**으로 둔다. 이는 사용자가 직접 제시한 숫자가 아니며 완료 보장도 아니다. 추가 profile·계수탐색 없이 기존 측정을 사용한다. 완료한 작업은 실제 시간으로 기록하고, 상한 시점에는 검증된 checkpoint와 완료 bank를 보존한다.

P672/813, Q2027/5097, V2026/5047과 원래 역할·소비 이력을 유지한다. Q는 private_train이자 이전 selection/Rwide 소비 집단이며 새 public/validation이 아니다. Q의 source 특징은 한 번 추출하고 모든 관련 arm에 재사용한다. 입력 영상·patient/class 분모·P+Q 합산 후 정규화·Q mixed102의 seed101 derangement를 검산한다. D는 제한 전 평균만 불변이며 제한 후 평균과2차 모멘트 모두 달라질 수 있다. Private raw feature·patient ID·목표 모멘트는 내부 저장물이다.

평가 수신자는 **DenseNet121 IMAGENET1K_V1**로 고정한다. 기존 P-only PCA128와 tensor 전처리를 유지하고, raw 관계 중심·scale도 이미 저장된 P 특징과 공개 mixed3명만으로 고정한다. 이 계산에는 수신 모델 성능을 사용하지 않는다. 수신자의 학습 입력은 최종 PNG·label·virtual pair·공개 학습 규칙이며 Q 특징/원자료를 받지 않는다.

네 bank의 최종 artifact hash를 모두 봉인하기 전에는 V·수신자 효용을 계산하지 않는다. 봉인 후 같은 V에서 A/B/C/D AUROC/AP를 계산하고, C/D의 동일 산출물 beta0 및 공개 실자료 점별 readout을 해석 보조로 함께 계산한다. 이는 추가 bank나 HPO가 아니다. Source에서는 float→PNG 모멘트/기능 차이만 기록하고 source V 효용을 별도로 추가하지 않는다.

기존 patient-cluster bootstrap 2,000draw를 재사용하여 같은 환자의 모든 영상을 함께 재표집하고, 모든 비교에 같은 draw를 적용한다. 95% percentile 구간은 **이번 고정된 한 seed의 bank들에 조건부인 개발 환자 불확실성**이다. 합성 seed 변동/독립 확인/최종 성능 증거로 확대하지 않는다. AUROC는 주지표, AP는 보조이며 weak image label을 그대로 사용한다.

B−A, C−B, C−D를 주요 예비 비교로 보고한다. 기존 개발 투자 기준의 효과 크기(.01 AUROC, C−D 양수, B 대비 AP 비감소)는 단일 seed의 탐색적 신호 표시에만 사용하고,3seed 중2개 방향 조건은 이번에 평가하지 않는다. 양성이든 음성이든 추가 bank는 자동 실행하지 않는다. 강한 관계 대조·순서 대조를 아직 하지 않았으므로 새 연산/선행 대비 우위도 확정하지 않는다.

ResNet18 반복, ViT 확인 수신자, E/E_R/R_joint, 보강 제거9bank, 추가 seed, DP, Expert532/810, Reserved4213은 이번 실행에서 제외한다. Expert/Reserved는 실제 pixel allow-list 밖에 유지한다. 원자료/비DP 산출물의 원격 업로드도 하지 않는다.

착수 전 실행기 마지막 검사에 남아 있던 microbatch4 전용 제한을4/8/16 허용으로 수정한다. 이는 앞선 속도 측정과 metadata 전달 검사만으로 발견하지 못한 연결 누락이었다. 해당 한 줄의 변경을 기존 파일과 대조하고 이미 완료한 실제 저장·재개·PNG 기술 검증은 반복하지 않는다. 기존 checkpoint/export/gradient/optimizer 연산은 바꾸지 않는다.

기계 판독 계약·코드/모델/manifest/projection 해시·범위 권한은 code_working/_reports/prrd_pilot_abcd101_20260921_v1/pilot_contract.json과 phase별 receipt에 결속한다. 결과 보고에서는 기술적 실행 정합성과 실제 효용을 구분하고, 네 bank 전체 수치·환자 조건부 구간·AP·공개 기준·PNG 오차·실제 시간을 함께 남긴다.

