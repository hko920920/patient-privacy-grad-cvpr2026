# 2-2 실행 계약: teacher-head 재구성 대조

2026-09-28. 사용자에게 실행 전 예상20~40분을 안내했다. 총 상한60분. 연구 단계2 유지.

질문: 현재 MT의 KD 대비 이득은 DINO head에 없는 class 공통 평균 정보를 유지해야만 가능한가? 일반 KD와 여러 공개 연산 차이가 함께 있었으므로 같은 지도를 허용한 head-only 대조를 만든다. 이는 내부 연산 대조이며 POST/KME/KIP 전체 재현 또는 신규 제안법이 아니다.

## 고정 설계

기존 공개128, 기존 DP1/DP2에서 파생된 DINO head, 공개 P의 DINO/RN 특징·가중치·지도·P-only target을 재사용한다. d=H_D w_D로 class 차이를 복구하고, s_P=(muP0+muP1)/2로 class 공통 평균을 고정한다. mu0'=s_P-d, mu1'=s_P+d를 기존 class별 affine 지도에 통과시켜 RN 목표를 만든다. pseudo 평균을 다시 clipping하지 않는다. 보호 class 공통 평균은 사용하지 않는다.

현재 MT와 비교해 바꾸는 것은 s를 s_P로 대체하는 것뿐이다. 지도·가중치·ridge0.1·zero-anchor·bounds[-1,1]·eta=.001||A||F²/128·RMS 규칙·BVLS를 유지한다. RN-only 라벨2묶음만 계산하며 Joint는 추가하지 않는다. 두 라벨을 모두 봉인한 뒤 V 평가한다.

새 readout: 2labels × DenseNet/ResNet18 =4. 기존1-5의20개 readout/예측/bootstrap은 그대로 재사용한다. DenseNet 영상AUROC 주평가, AP 보조, RN은 구성 모델 참고치. 기존2000회 환자-cluster bootstrap을 동일하게 사용한다.

## 비교와 사전 판정

1. 주 비교 MT RN-only−HR: 보호 class 공통 평균의 추가 가치.
2. 보조 HR−KD RN-only: 같은 DINO head 정보의 공개 처리 차이.
3. 보조 HR−DINO-only: 목표 전달의 강한 단독 대안 대비 가치.

개발 투자 hurdle은 기존 규칙 그대로: 두 release AUROC 점차이 모두 양수, 평균AUROC≥0.01, 평균 조건부95%CI 하한>0, 평균AP≥0, release별 유의AP악화 없음. 0.01은 임상적 보편 기준이 아니다. MT−HR가 이 기준을 못 넘으면 추가 s의 필요성은 미확인으로 둔다. 작은 점차이나 CI0 포함을 동등성으로 바꾸지 않는다.

HR−KD가 양성이면 현재 teacher-score KD 비교의 범위를 좁힌다. 같은 정보의 처리만으로 이득을 얻을 수 있다는 개발 근거이며, 기전의 유일 원인이나 선행 전체 우위의 증거는 아니다.

다음2-3은 MT 또는 HR가 DINO-only 대비 투자 hurdle을 충족할 때만 실행한다. MT는1-5에서 미충족한 상태를 그대로 유지한다. HR가 통과하더라도 자동으로 새로운 방법/기여로 승격하지 않고 후보/대조의 한 공개선정 반복으로만 진행한다. 둘 다 미충족이면2-3과3-2는 보류하고 그 이유를 보고한다. 좋은 결과가 나올 때까지 라벨·계수·encoder·이미지선정을 바꾸지 않는다.

## 실행 경계

새Q access·DP noise/release·pixel학습·encoder forward·수신자탐색·Expert/Reserved 없음. 기존 P-only target은 공개자료만을 사용한 것으로 계약에 결속한다. 기존 보호 head의 후처리이며 추가 개인정보 비용0. 환자 예측은 로컬 평가 artifact에 보존한다. 입력/코드/이전 결과 해시를 결속한다. 변경된 목표 경로의 head 복구·지도 사용·라벨KKT·ridge·같은 bootstrap 대비만 검산하며 전체 과거 pipeline을 다시 실행하지 않는다.
