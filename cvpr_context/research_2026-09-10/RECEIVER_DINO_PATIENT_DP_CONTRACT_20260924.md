# DINO 단독 환자-DP 대조 실행 계약

2026-09-24 KST. 연구 단계2의 개발 비교다. 사용자 최신 지시로 DINO 전용 보호 경로 준비·검산, 독립 보호 요약1회, 합성 bank1개와 개발 평가를 수행한다. 기존 A2 DP 잡음1·2 결과를 모두 재사용한다.

## 질문과 범위

같은 환자-DP 예산에서 두 source 결합의 효용이 DINO 단독보다도 남는가? DINO 단독에도 자기 query와 공개 clipping 기준에서 전체 ε8/δ10⁻⁵을 사용하도록 한다. A2 보호 요약의 DINO 좌표만 잘라내지 않는다.

- P 공개672명/813장, Q 보호2027명/5097장, V 기존 개발2026명/5047장. 기존 source·head·K4 조건·P-only PCA16/q95와 class별 환자 균등 가중치를 유지한다.
- DINOv2 ViT-B14 단독. 기존 seed101 공개 초기 template·optimizer·해상도 일정,128장·200회·microbatch16. K4 cosine 평균의 단일-source 가중치1, anchor/TV 한 번. 마지막200회 PNG만 평가한다.
- 새 특징 추출, profile, 계수 탐색, 추가 seed/noise, Expert/Reserved/확인용 수신자 접근은 하지 않는다. 기존 비DP DINO 및 A2 두 DP bank를 재학습하지 않는다.

## 보호 query

각 환자 p/class c에서 방문별 DINO gradient의 평균 g_pc를 만든다. K4×34=136좌표다. 두 class의 공개 P 환자 norm 분포에서 각각 q95(linear quantile, floor10⁻⁶)로 C_c를 정하고, Q 값을 읽기 전에 동결한다. P의 양성 class 환자는6명이며 공개 선택이 최적성을 보장하지 않는다.

v_pc=clip(g_pc,C_c), b_pc=class 방문 존재 표시로 두고,

u_p=0.5[v_p0/C_0; v_p1/C_1; b_p0; b_p1].

전체274좌표이며 ||u_p||²≤0.25(1+1+1+1)=1이다. 한 환자의 모든 영상과 class/head/condition이 하나의 보호 단위다. Q 합계의 add/remove-one-patient L2 감도는1이다. 보호되지 않은 Q 환자 수를 고정 분모로 사용하지 않는다.

전체 벡터에 ε8/δ10⁻⁵ analytic Gaussian을 한 번 적용한다. 검증된 A2와 동일한 감도1 보정 σ≈0.6002290723, 보수적1e−10 margin. 좌표 수와 C_c는 DINO 전용이다. OS 난수로 두 독립 normalvariate의 합/√2를 사용하며 재현 가능한 DP seed·실현 잡음을 저장하지 않는다. 수치 검산은 별도의 유한정밀도 보안 증명이 아니다.

후처리는 A2와 동일하다. 두 noisy class mass를0 이상으로 제한하고, noisy sum을 C_c×mass 반경에 투영한다. 공개 P의 sum/count와 결합해 class별 pooled 평균을 만든 뒤 두 class에0.5씩 가중한다. 분모 하한1, 목표가0이면 사전 P 신호 fallback. 새 noise를 다시 뽑지 않는다. P 신호 자체는 clipping하지 않는다.

## 변경 경로 검증

구성 배열에서 환자/class 가중치, 반복 방문,274좌표 경계·감도, 빈 class/Q, 음수 noisy mass, 잘못된 입력, 잡음 좌표 수를 확인한다. 실제 저장 P/Q DINO 특징에서 기존 sum/count/무제한 무잡음 pooled target 재현오차≤10⁻¹²를 확인한다. 공개 모의 target의 직렬화·feature gradient 일치도 확인한다. 이미 검증된 DINO의 renderer/two-pass/optimizer/저장·재개를 그대로 재사용하며 전체 GPU 검사를 반복하지 않는다.

## 평가와 해석

최종 PNG를 고정한 뒤 같은 BioViL·DenseNet ridge readout, 기존 V 특징과2000회 paired 환자 bootstrap을 재사용한다. 주 비교는 각각 A2 DP1−DINO DP, A2 DP2−DINO DP의 DenseNet AUROC이며 AP를 함께 보고한다. 양쪽 모두 ΔAUROC>0, 조건부 환자95% CI 하한>0, ΔAP≥0이면 ‘두 고정 비교에서 A2 이득’을 기록하고, 그 외에는 혼합/명확한 우위 미확인으로 기록한다. 이는 사전 기술적 해석 규칙이며 논문 유의성·잡음 분포 전체의 우위를 뜻하지 않는다. 좋은 A2 잡음 하나만 선택하지 않는다.

동일 업데이트 수이며 동일 계산비용 비교는 아니다. DINO는 새로운 독립 잡음 한 번, A2는 과거 두 번이다. DenseNet/V는 개발에 재사용됐고, 환자 bootstrap은 bank·잡음을 고정한 구간이다. BioViL은 A2의 source이며 DINO 단독 제작에는 사용되지 않는다. 기존 A2 공개전용 결과는 맥락용이며 DINO 전용 공개 대조로 바꿔 부르지 않는다.

## 시간·저장·공개 경계

예상 전체50~70분(기존 DINO 합성·저장34.71분 참고). bank 누적60분, 전체90분 상한: 시작2026-09-23 15:38:08 UTC, 마감17:08:08 UTC. 시간·데이터·수치 검증 실패 시 정지하고 검증된 checkpoint만 재개한다. 결과를 보고 자동 재실행하지 않는다.

새 요약은 DINO_DP8_release01로 기록한다. 각 A2/DINO 실행은 개별(8,10⁻⁵) 설정이며, 같은 Q에서 만든 기존 A2 두 개와 새 DINO를 모두 공개하면 기본 합성의 보수적 상한은(24,3×10⁻⁵)이다. A2 두 개만의 기존 상한(16,2×10⁻⁵)은 그대로다. [Dwork–Roth, Theorem3.16](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

보호 범위는 고정된 공개 recipe에서 나온 Q 요약과 후처리다. 과거 비DP 개발 선택, 내부 Q 진단, V 평가 보고서 전체까지 보호하지 않는다. 실제 외부 공개는 수행하지 않는다. 보호 PNG 묶음과 내부 감사파일을 분리한다. 이번 bank의 중복 checkpoint만 기존 규칙대로 정리하며 이전 결과·cache는 보존한다. 정상 구간은 목표 준비→학습→저장→평가까지 이어서 진행하며 추가 실행은 예약하지 않는다.
