# Feature 공개전용 및 독립 DP 잡음 반복 — 실행 계약 (2026-09-24)

사용자의 “실행” 지시로 공개전용 1bank와 독립 DP 잡음2 1bank, 총 2bank를 승인된 한 묶음으로 실행한다. 단계2 개발 비교이며 CVPR 기여나 일반적인 복잡도 원리를 확정하는 실험이 아니다.

- 기준은 93a4067 feature-mean/L2 구성과 완료된 feature DP1이다. 동결된 DINO checkpoint, P-only PCA16/q95, 네 개별 변환 조건, class/환자 가중치, 공개 clipping 경계, loss, 공개 template와 초기 상태를 재사용한다.
- 각 bank는 seed101·128장·200회·microbatch16·동일 AdamW·해상도 일정으로 새로 시작한다. 이전 학습 checkpoint는 초기 상태 검증에만 쓰고 이어 학습하지 않는다. 마지막200회 PNG만 채택한다.
- 공개전용 target은 기존 bounded P class/condition 합계를 P class 환자수669/6으로 나눈다. 기존 public_fixture와 수치가 정확히 일치해야 한다. Q 통계나 분모를 섞지 않으며 protected-query 수0으로 기록한다.
- DP2는 기존과 동일한 130좌표 query, 모든 방문/class/네 조건을 포함하는 add/remove 환자 인접성, sensitivity1, ε8/δ1e-5, analytic Gaussian sigma0.6002290722589745를 유지한다. 공개 C0/C1은 기존 calibration 파일의 정확한 값을 재사용한다. 비공개 class 분모 처리·consistency projection·public denominator floor는 변경하지 않는다.
- 새로운 OS-random Gaussian draw는 정확히 한 번이다. random coins, 실제 noise, unnoised query를 저장하지 않는다. 보호된 noisy query와 목표만 저장한다. 예약 후 실패 시 재추출하지 않으며 보존된 noisy query가 있으면 동일 release의 후처리만 안전하게 재개할 수 있다.
- 두 최종 PNG가 모두 고정되기 전에는 새 수신자 성능을 보지 않는다. 기존 DenseNet(주개발), ResNet18(추가개발)의 V 특징·공개 projection·readout 및2000회 동일 환자 bootstrap을 재사용한다. 기존 Feature DP1·DINO gradient DP1/DP2 결과는 재학습·재평가하지 않는다.
- 필수 대조는 각 Feature DP1/DP2−Feature 공개전용, 각 Feature DP1/DP2−각 Gradient DP1/DP2, Feature DP2−DP1이다. AUROC/AP 점추정과 고정 bank에 조건부인95% 환자 구간을 모두 보고한다. 두 잡음의 기술통계는 앙상블·모집단 분산 추정·동등성 증거로 해석하지 않는다.
- Q 추가효용과 두 번째 잡음 관측이 질문이다. 우열·동등성·정보량 단조성·DP interaction·CVPR 기여를 자동 판정하지 않는다. 결과에 따라 목표/계수/잡음/학습량/checkpoint를 변경하지 않는다.
- 이미 통과한 이미지 gradient·저장·재개 검사를 재실행하지 않는다. 변경된 P target 선택, population/DP metadata 분리, loader 동일성과 잘못된 job 거절만 추가 검사한다. 새 Q/V 픽셀 추출0; 현재 cache 재사용.
- 합성 예상 약70분, 준비·평가 포함90–110분. 전체 상한120분: 시작08:30:40UTC, 종료10:30:40UTC. bank당 누적3600초. 시간 상한·수치검증 실패·안전한 재개 불가·설계 변경 필요 시 중단한다. 저장공간1GiB 미만이면 안전 중단한다.
- 예상 main DINO image forwards409600 / backwards204800, 새 PNG 평가512 forward. 추가 profile·계수 탐색 없음.
- 새 private release1개, 기존5개와 공동 공개 시 basic composition 상한(48,6e-5). 공개전용 bank는 추가 비용0. Q에 대한 보호 경계이며 비DP 개발·V 평가·선택 기록 전체를 보호하지 않는다. 외부 공개/업로드는 수행하지 않는다.
- 새 평가 수신자·추가 seed·DP3·학습 연장·A2 재학습·Expert·Reserved·확인용 수신자·별도 interaction 분석으로 자동 확대하지 않는다. 본 두-bank 묶음을 끝내고 비교 결과와 실제 비용을 보고한다.

