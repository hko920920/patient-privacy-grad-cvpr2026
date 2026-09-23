# DINO/A2 seed202 대응 반복 — 2026-09-23

큰 단계2의 개발 반복 확인이다. seed101에서 관측한 A2−DINO 결합 이득이 새 합성 초기화에서도 남는지 확인한다. 이번 범위는 DINO 단독/A2의 두 bank와 두 최종PNG 동결 뒤 동일 개발 평가까지다.

## 고정 범위

- DINO, A2 각각128장·200 update·microbatch16. AdamW 및 pyramid 활성1/33/65/97/129/161 유지.
- synthesis seed202와 target condition/head seed101을 별도 인자로 기록한다.
- 기존 공개 template 선택 정책의 seed202 목록을 재사용한다. 두 방법의 template·초기 residual·optimizer·난수·활성상태가 같다.
- 모델·P-only projection·K4 affine/head·환자/class 가중 P/Q 목표는 seed101의 실제 파일을 그대로 쓴다. 새 목표 추출 없음.
- DINO 단독은 DINO 손실1, A2는 BioViL0.5+DINO0.5. 각 모델의 K4 평균과 anchor0.01/TV0.0001 한 번 적용은 그대로다.
- 모델·목표·head 난수 변화에 대한 반복이 아니라 공개 template 선택 및 합성 초기 residual을 포함한 초기화 반복이다.
- 기존 seed101 코드/산출물은 변경하지 않는다. 명시적 --seed202를 받는 별도 실행 모듈을 사용한다.
- 원래 seed101과 동일한 초기화 및 모든 해상도 출력, seed202의 다른 초기 residual, 대응 초기 상태, seed가 다른 재개 거부, 기존 objective digest 보존을 검사한다. 기존 model gradient·저장/재개·PNG 검증은 재사용한다.

## 비용·실행

실측 참고: DINO34.71분 + A2 73.92분 = 합성 약109분. 연결·검사·평가·기록 포함 사전 예상2–2.5시간. 시간 상한은 DINO worker60분/A2 worker120분이며 시작·저장·최종PNG까지 포함한 누적 시간이다. 준비·평가는 별도 기록한다.

DINO → A2 → 두 최종PNG seal → 동일 V 평가를 이어서 수행한다. 정상 구간마다 다시 승인받지 않는다. 초기 상태 비교는 실제 두 bank의 renderer·optimizer·RNG·활성상태 해시로 수행한다.

추가 profile, 계수 탐색, seed303, A1 재학습, 500회 연장, DP, Expert, Reserved, final receiver는 실행하지 않는다. 수치/자료 검증 실패, 안전한 재개 불가, 시간 상한에서 중단한다.

## 평가

기존 BioViL/DenseNet V 특징·ridge0.1 readout·2,000회 paired 환자 bootstrap을 재사용한다. 두 bank를 모두 고정하기 전에는 효용 평가를 열지 않는다.

Primary 개발 비교는 seed202의 DenseNet A2−DINO AUROC, 함께 AP와 BioViL AUROC/AP 및 비용을 보고한다. seed101 결과도 그대로 옆에 둔다. 새 절대 통과선이나 결과 기반 계수 선택을 추가하지 않는다.

각 seed의 bootstrap 구간은 해당 두 bank에 조건부인 환자 표집 불확실성이다. 두 seed 차이의 단순 평균은 설명용이며 모집단의 합성 seed 불확실성 구간이나 예측 ensemble 성능으로 부르지 않는다. DenseNet과 V는 적응적으로 사용된 개발 자원이다. BioViL은 A2에 사용된 source이므로 독립 unseen receiver로 세지 않는다.

결과에 관계없이 이 묶음 이후 자동 확대하지 않는다. 사적 자료의 추가가치·DP·독립 receiver·동일 계산량의 우위는 별도 질문이다.

## 파일

code_working/_reports/receiver_repeat202_s200_20260923_v1/에 기계 판독 계약, seed 검사, 실제 코드/목표 결속, 두 bank·로그·평가를 저장한다. 중복 checkpoint는 이번 bank 안에서만 정리하고 step0·마지막두/final 및 모든 receipt/trace를 남긴다.

모든 Q 유래 목표와 합성자료는 비DP 내부 연구 산출물이다.

