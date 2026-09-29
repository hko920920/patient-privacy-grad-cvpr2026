# 공개128에서 일반 KD 대조 — 2026-09-28

큰 단계2의 제한된 개발 비교다. 사용자 ‘이어서 쭉 하자고’에 따라, 1번에서 관측한 전달 방식의 이득이 3번의 고정 공개128에서도 남는지 확인한다. 이전 결과와 실패는 보존한다.

- 기존 feature-DP1/DP2, 공개128 선정·이미지 순서, DINO/RN 특징·공개 가중치·목표를 재사용한다.
- 현재 평균 전달(MT)의 RN-only/Joint 라벨 네 묶음과 평가는 3번 것을 재사용한다.
- 일반 teacher-score KD의 RN-only/Joint × DP1/DP2 라벨 네 묶음만 새로 계산한다. 가장 유리한 recipe만 선택하지 않는다.
- KD 목표는 1번과 같다: q=X_D w_D, w_R=(X_R'WX_R+0.1I)^(-1)X_R'Wq. sigmoid·temperature·outcome tuning 없음. 1번의 봉인된 목표 계수를 사용하고 공개 입력의 별도 least-squares 표현으로 검산한다.
- 두 목표 방식 모두 3번의 zero-anchor 목적, bounds[-1,1], ridge0.1, eta=0.001||A||_F^2/128, 공개 목표 RMS 정규화, BVLS tol1e-12/max_iter500을 사용한다. 규칙이 같아도 목표 RMS와 eta 숫자는 달라질 수 있다.
- RN-only/Joint 두 층에서 별도로 판정한다. DenseNet 영상 AUROC 주 지표, AP 필수 보조 지표. ResNet18은 목표 생성 모델의 참고 평가다.
- 기존 2,000회 같은 환자-cluster bootstrap. 각 release의 MT−KD와 두 release의 차이 평균을 같은 bootstrap 행에서 계산한다. 조건부 구간이며 noise 모집단·독립 검증·동등성 주장이 아니다.
- 기존 투자 기준: 두 AUROC 차이 양수, 평균 차이≥0.01, 평균 AP≥0, 평균 AUROC 조건부95%CI 하한>0. 유의한 AP 악화가 있으면 혼합 결과로 표시한다. 각 recipe 별도 판정, 둘 중 좋은 결과로 일반화하지 않는다.
- MT 점수는 이미 본 적 있으므로 사후 개발 비교다. 새 KD 점수 확인 전 계약·네 라벨을 고정한다. 성능에 따라 재설정·재시도하지 않는다.
- 새 Q 접근·잡음·DP 요약·pixel 최적화·모델 forward·공개영상 재선정·새 수신자·Expert/Reserved 접근 없음. 추가 Q 회계0. 공개128+DP 라벨은 public-only 대조가 아니다.
- 예상 총20~40분, 작업 상한60분, 수치 단계별600초. 저장 목표10MiB 이하/상한50MiB. 기존 모델·영상·캐시 복사 없음.
- 분석 산출물: contract, 새 라벨4개, 재사용MT 라벨/점수 연결, paired 비교표, 수치 검산, 결과·상태 기록. 이 완료를 CVPR 기여 확보로 표시하지 않는다.
- 이번 비교 후 새 bank/release/독립 평가를 자동 추가하지 않는다. 다음 과학적 판단을 보고한다.
