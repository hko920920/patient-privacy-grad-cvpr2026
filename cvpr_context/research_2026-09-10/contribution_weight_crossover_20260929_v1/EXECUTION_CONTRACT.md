# 2-2R2: 공개 대응 가중치 교차 대조

사용자 진행 승인 후 시작. 시작 2026-09-29T06:27:27Z. 실행 전 예상20~30분을 안내했다. 상한45분.

목적: 질환용 class-balanced 학습과 공개 모델 대응용 가중치를 분리하면 현재 전이 이득을 설명하는지 확인한다.
2-1R2의 NEXT_2_2R2_DESIGN.json을 그대로 따른다. 최초 공개128과 기존DP1/DP2 사용.

- HR에 B(class-balanced), ACKD에 U(patient-equal) 대응 가중치를 적용한 새4labels와8readouts.
- 기존 HR-U, ACKD-B, DINO-only의12readouts를 재사용.
- 기존 공개 midpoint, source/receiver task Hessians, task class weights, fixed numerical lambdas, 공개128, RN-only label solver를 고정.
- HR의 fitted means/intercepts는 대응 가중치 변경의 결과로 함께 바뀐다. slope-only 대조는 아니다.
- 모든 새라벨/목표/코드를 봉인한 뒤 V를 한 번 평가. 두 주 대비와 지정한 보조 대비를 전부 보고.
- 새Q/noise/release/영상학습/model forward/Expert/Reserved/새수신자/계수검색 없음.
- 두 주 대비 모두 기존 engineering hurdle을 만족할 때만 공통 가중치 가설을 강하게 지지.
- 결과에 따른 반복·대조 추가·독립 평가로 자동 확대하지 않음.
