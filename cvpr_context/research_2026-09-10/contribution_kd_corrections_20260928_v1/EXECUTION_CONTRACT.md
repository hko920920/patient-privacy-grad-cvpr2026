# 2-2 보완(2-2R1) 실행 계약

실행 전 사용자에게 **20~35분**을 안내했다. 2-1R1의 수식/근접연구 설계를 그대로 사용한다. 전체60분, 수치 phase600초 상한. 기존 2-3 보류 gate는 바꾸지 않는다.

## 질문과 범위

같은 DINO head와 공개자료만 사용하는 HR가, 공개 기준값 보존과 공분산 보정을 허용한 KD보다도 추가로 유용한가?

기존 KD를 2×2의 한 칸으로 재사용하고 새 AKD / CKD / ACKD를 각각 기존DP1/DP2에서 한 번씩만 계산한다. 동일 공개128장·RN-only label solve·zero anchor·bounded [-1,1]·ridge0.1. 6labels,12readouts,기존24readouts 재사용. 새 Q/noise/release/image optimization/model forward/newreceiver/Expert/Reserved0.

## 고정 수식

D=P DINO64, R=P RN128, Ω=기존 class/patient/visit 가중.
G=D'ΩD, H_D=G+0.1I, H_R=R'ΩR+0.1I, C=R'ΩD.
P의 clipped class 목표로 d_P=(mu_P1-mu_P0)/2, w_DP=H_D^-1 d_P.
RN 공개 supervised 기준 w_RP=H_R^-1 R'Ω(2y_P-1).
λ_C=0.001 trace(G)/64.

A_K=H_R^-1 C.
A_C=H_R^-1 C (G+λ_C I)^-1 H_D.

- KD=A_K w_D: 기존 결과 재사용.
- AKD=w_RP+A_K(w_D-w_DP).
- CKD=A_C w_D.
- ACKD=w_RP+A_C(w_D-w_DP).

λ는 공개 trace 규칙 하나다. sweep/성능 재조정하지 않는다. 각 목표는 독립 public pseudo-response ridge로 검산한다. source 기준값은 보호신호 schema의 P목표이고 RN기준값은 실제 P labels를 사용한다.

## 라벨과 평가

검증된 solve 함수를 그대로 호출한다. 같은 normalization/eta규칙,[-1,1],BVLS. 새6labels 모두 봉인 후 V 캐시 평가. DenseNet 주개발 receiver, RN은 구성 모델 참고치. 같은2000회 환자-cluster paired bootstrap. 기존24scores/bootstrap을 bitwise 재사용. 예측값은 로컬 내부 artifact이며 외부 게시하지 않는다.

## 판정

주대조 HR−ACKD. 함께 HR−AKD, HR−CKD를 보고한다.
2×2 요소 대조 AKD−KD, CKD−KD, ACKD−AKD, ACKD−CKD와 조건부 interaction을 보고한다.
새3종−DINO-only도 모두 보고한다.

개발 hurdle: 두release AUROC점차이>0,평균차이>=0.01,평균조건부환자CI하한>0,평균AP>=0,개별release AP CI상한<0 없음.
HR의 추가 차별효용을 다음 투자 근거로 삼으려면 세 새 보정 각각에 대해 이 기준을 만족해야 한다.
구간이0을 포함한다고 동등성/원인설명완료를 선언하지 않는다.
새 표준보정이 잘되면 알려진 보정의 유용성으로 기록하며 새발명으로명명하지 않는다.

모든 interval은 현재 fixed releases/carrier/banks에 조건부다. 여러 보조검정은 확증적 familywise 주장으로 쓰지 않는다. 두noise결과로 expected method ranking이나 noise robustness를 주장하지 않는다.
원래2-3의MT/HR source-only gate는계속보존한다. 결과에따른새후속은새질문/설계/ETA로보고하며자동추가release나독립평가를하지않는다.

