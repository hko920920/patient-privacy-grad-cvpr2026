# 1번 단독 실행 계약 — 2026-09-28

사용자가 예상 30~60분 안내 후 “진행”으로 승인했다. 큰 단계2의 직접 대조이며, 전체 1·2·3 계획 중 **1번만** 실행한다.

질문: 동일한 두 feature-DP 요약과 PNG에서 현재 평균 전달(MT)이 일반 teacher-score KD보다 유용한 ResNet18 목표를 만드는가?

- DP1/DP2 둘 다 사용. 새로운 라벨 대조는 KD 두 묶음뿐.
- 기존 MT 라벨·PNG·평가 결과를 보존/재사용. MT 라벨 계산 2회는 경로 재현 검사이며 새 방법 후보가 아니다.
- 기존 공개 P, DINO K4 projection, ResNet18 공개 특징, patient/class weighting, ridge 0.1 유지.
- DINO teacher w_D=(X'WX+0.1I)^(-1)(mu1-mu0)/2; q=Xw_D.
- KD receiver w_R=(Y'WY+0.1I)^(-1)Y'Wq. q를 clamp/sigmoid하지 않음.
- 이후 joint DINO/RN label solve는 기존과 동일: 모델별 공개 target RMS 정규화, 각0.5, eta=.001||A||F^2/128, 원래 64개 -1/64개 +1 anchor, [-1,1] bounded labels, tol1e-12/max_iter500.
- **이번에는 zero-centered regularizer로 바꾸지 않는다.** 그것은 공개 carrier 비교를 위한 전체 계획이며 1번만 하는 현재 범위에는 불필요하다.
- 동일 숫자 eta를 강제하는 것이 아니라 동일 산출 공식을 유지한다. RN target/RMS가 달라지므로 실제 eta·scale은 기록한다.
- 공개128 선정, source-only 신규 반복, nullspace 탐색, 다른 encoder, 새 요약/noise, pixel optimization, Expert/Reserved, full-network 학습 없음.
- 두 KD 라벨을 모두 동결/hash한 뒤 evaluator 실행. construction에 V 데이터/지표 접근 없음.
- DenseNet 개발 AUROC primary, AP secondary. RN은 construction 모델이라 진단/참고 지표다.
- 기존 영상별 AUROC/AP + 2,000 paired patient-cluster bootstrap 재사용. 동일 release의 MT−KD 비교 및 두 차이의 평균을 보고한다. 평균은 score ensemble이 아니다.
- 각 구간은 현재 두 bank/release와 개발 평가에 조건부다. 새로운 독립 확인 또는 선행 전체 재현 우위로 주장하지 않는다.
- 다음 투자 참고: 두 AUROC 차이>0, 평균≥.01, 평균AP≥0, 평균 AUROC 조건부 CI하한>0일 때 추가 가치의 개발 근거. 방향/CI가 혼합이면 그대로 기록하고 자동 확장하지 않음. 차이 불확실을 동등성으로 취급하지 않음.
- KD normal equation을 독립 augmented least squares와 비교. MT 원래 라벨 재현, primal/dual readout, bootstrap3개 draw의 sklearn 독립 검산.
- 수치 기준: KD/MT 계수 차이1e-10, MT 라벨 재현1e-8, readout score차이1e-10, bootstrap검산1e-12, projected-gradient≤1e-7. 실패시 허용오차나 recipe를 결과에 맞춰 바꾸지 않음.
- 예상 전체30~60분. 실제 numerical worker당600초 한도, 전체 작업60분 목표. 결과와 별도로 실시간을 기록한다.
- 기존 여섯 DP release 공동 회계48/6e-5 유지, 이번 Q 추가 회계0. 결과물은 각 원래 feature release의 후처리이며 과거 개발 과정 전체를 소급 보호하지 않는다.
- 모델·데이터·checkpoint 다운로드/복사 없음. 기존 PNG는 참조+SHA만. 추가파일250MiB 이내 목표.

