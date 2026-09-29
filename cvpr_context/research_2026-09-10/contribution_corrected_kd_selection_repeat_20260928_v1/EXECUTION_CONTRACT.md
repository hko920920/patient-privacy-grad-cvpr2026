# 2-3 보완(2-3R1) 실행 계약

실행 전 **20~35분**을 보고했다. 전체60분, 수치phase600초 상한.
2-2R1의 강화KD대조 통과에 따른 **공개carrier 선정 반복**이다. 기존2-3의source-only gate는유지하고소급통과시키지않는다.

## 변경과 규모

기존 public128 알고리즘을 그대로 호출하고 salt만 `public-carrier-20260928-v1-repeat202`로 바꾼다.
672 P환자별1영상후보, 양성6환자 포함, DINO/RN equal-RMS 공간의 farthest-first 128선정.
Ponly. DP목표/DenseNet/V로선정하지않는다. 1회만고른다. 이전선정과겹침을보고한다.

목표는기존봉인파일그대로:
HR / AKD / CKD / ACKD / DINO-only × DP1/DP2=10labels.
source-only는 DINO 목표만, 나머지는RN목표만같은zero-anchor[-1,1]solver로맞춘다.
새라벨봉인후 DenseNet/RN20readouts. 기존36개는고정참고결과로재사용.
이번작업에서전달지도나head를다시맞추지않는다.
Q/noise/release/pixeloptimization/model-forward/newreceiver/Expert/Reserved0.

## 비교와 판정

주비교HR−ACKD. HR−AKD/CKD와 HR−DINO를동시보고.
기존투자hurdle그대로: 두release AUROC점차이>0,평균>=0.01,평균조건부CI하한>0,평균AP>=0,개별release AP유의악화없음.
강화대조재현판정은HR가세KD대안각각에 대해새선정에서도이기준을통과하는가.
source-only판정은별개로보고하고기존선정의DP2음수결과를삭제하지않는다.

같은기존DP2개/같은P/V의부분선정변경이다. 독립환자집단,새DP잡음,untouchedreceiver검증이아니다.
영상AUROC/AP,동일2000회환자cluster bootstrap. 평균은지표차이의산술평균이지ensemble이아니다.
두선정전체를묶어독립4회재현이라고세지않는다.
어느결과든추가salt/seed/receiver를자동탐색하지않는다.

