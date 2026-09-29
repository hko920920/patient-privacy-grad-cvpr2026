# 1-5 실행 계약 — 공개128의 DINO-only 대조

2026-09-28. 사용자 요청에 따라 번호 순서대로 실행·결과 보고·다음 설계/규모/예상 시간 안내 후 이어간다. 상위 실험계획1과 연구 전체 단계2를 구분한다.

**실행 전 예상: 15~30분. 작업 상한45분, 수치 단계별600초.** 기록 시작17:04:48 KST. 이후 정상 작업 사이에 승인 질문을 반복하지 않는다.

- 고정 공개128·기존 DP1/DP2의 DINO 목표로 DINO-only 라벨2개만 새로 계산한다.
- 1-4에 봉인된 MT RN-only/Joint와 KD RN-only/Joint의 라벨·점수·bootstrap은 재사용한다.
- 3번/1-4와 같은 zero-anchor, bounds[-1,1], ridge0.1, eta=.001||A||_F²/128, P 가중 목표 RMS, BVLS tol1e-12/max_iter500. 단일 DINO 모델 비중1.
- 공개 P의 class별 환자 균등 가중, 조건·features·공개128 선정·모든 전처리와 readout 규칙을 유지한다.
- 주 비교: MT RN-only−DINO-only, MT Joint−DINO-only. KD 두 구성−DINO-only는 참고 대비. 비교마다 동일 DP1/DP2를 대응시킨다.
- DenseNet 영상 AUROC 주 지표/AP 보조, RN 참고. 같은2,000회 patient-cluster draw. 고정 release/라벨 조건부 CI이며 adaptive development 및 noise 분포 전체 불확실성을 포함하지 않는다.
- 기존 개발 기준 유지: 두 release AUROC 차이 양수, 평균≥.01, 평균AP≥0, 평균AUROC CI하한>0. 유의한 AP 악화가 있으면 혼합 판정. 좋은 recipe만 고르지 않는다.
- 새 두 라벨을 봉인한 후 평가한다. 결과에 따른 라벨 재시도·계수 조정·이미지 재선정 없음.
- 새 Q/noise/release/pixel/encoder forward/receiver/Expert/Reserved 접근0. 기존 추가 Q 회계0. 원안 잔여 합성4칸/nullspace 진단은 이번 범위가 아니다.
- 저장 목표10MiB 이하/상한50MiB. 기존 이미지·모델 복제 없음.
- 완료 후 번호1-5의 방법·결과·한계·실측 비용을 보고하고 상위1을 정리한다. 다음2-1은 먼저 설계·규모·예상 시간을 안내한 후 시작한다. 이 계약이 새 DP release나 final 평가를 자동 허용하지 않는다.
