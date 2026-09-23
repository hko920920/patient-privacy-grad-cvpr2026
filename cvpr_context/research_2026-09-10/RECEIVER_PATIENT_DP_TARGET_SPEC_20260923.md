# A2 환자-DP 목표 경로 — 실행 전 명세 (2026-09-23)

큰 연구 단계2의 구현·검산 작업이다. 이번에는 기존 캐시를 통한 비DP 재현·공개 기준 선정·구성 배열의 보호 연산 검사만 수행한다. Q에 잡음을 적용한 새 target/release, 합성 bank, V 효용, Expert/Reserved/확인용 수신자는 실행하지 않는다.

## 고정한 설계

P는 공개672명, Q의 보호 단위는 환자 한 명의 모든 영상/방문/class다. 두 source(BioViL, DINOv2), K4 조건/head seed101, P-only projection과 합성 목적/계수는 변경하지 않는다. 보호 인접성은 Q에 환자 한 명 전체 추가/제거다. 전체 Q 환자 수를 공개 상수 분모로 사용하지 않는다.

각 환자/class에서 기존 per-image CE gradient를 방문 평균하고, 두 encoder×네 조건×34 좌표를 이어 붙인 g_pc∈R^272와 class 존재 지표 b_pc를 만든다. class별 clipping 기준 C_c는 **P의 해당 class 환자 gradient norm95% 분위수(linear interpolation), 최소1e-6**로 고정한다. Q/V로 선택하지 않는다. class 양성이 적은 P의 기준이 최적이라는 주장은 하지 않는다.

v_pc = g_pc min(1, C_c/||g_pc||).
u_p = (1/2)[v_p0/C_0, v_p1/C_1, b_p0, b_p1] ∈ R^546.
각 v_pc는 class가 없으면0이다. ||u_p||²≤(1+1+1+1)/4=1.
합계 query Σ_Q u_p의 add/remove sensitivity는1, replace-one 상계는2다. 코드의 마지막 unit-ball 제한은 부동소수점 경계 초과를 막고 모든 좌표에 공통 적용한다. 정상 산술에서 class count는 그대로 유지된다. 이 추가 제한이 활성화되면 count 좌표도 가중 질량이 됨을 기록한다.

첫 보호 비교의 제안값은 **ε=8, δ=1e-5, one full-vector Gaussian query, no subsampling**이다. Balle–Wang analytic Gaussian calibration을 사용하고 Google dp_accounting 및 고정밀 식으로 독립 검산한다. 200 합성 update와 두 encoder×K4마다 개인정보 예산을 다시 쓰지 않는다. class count까지 한 query에 포함한 예산이다. 이전 PRRD 모멘트의 감도를 가져오지 않는다.

전체546좌표에 독립 표준편차 σ의 잡음을 더한다. gradient 합계의 원 단위 noise std는 class별2C_cσ, count noise std는2σ다. P의 합계/분모는 보호하지 않고 그대로 결합한다.

보호 후에는 Q count를 max(0,noisy_count)로 만들고, Q class gradient 합계를 반경 C_c×nonnegative_count의 ball로 투영한다. 기존 pooled 방식으로
t_mk = (1/2) Σ_c (S_P,mkc + S_hat_Q,mkc) / max(1, n_P,c + n_hat_Q,c)
를 계산한다. 분모와 일관성 투영은 보호된 수치+P만 사용하는 후처리다. clipping/noise를 끈 검산 경로는 기존 P+Q target을 그대로 재현해야 한다. Q=empty를 실제로 보호할 때도 잡음을 생략하지 않는다.

조건별 target의 norm≤1e-12인 경우에는 해당 공개 P target으로 대체한다. 이는 기존 cosine 정의의 수치 예외 처리이며 Q 원자료를 다시 보거나 noise를 다시 뽑지 않는다. NaN/Inf 및 구조 오류는 검산/사전 입력 검증에서 거부한다.

## 산출물 경계

내부: Q 원특징/환자 ID/원분모/합계/clipping 진단/캐시 hash/과거 비DP 비교.
향후 보호 handoff: 공개 규칙·공개 calibration·공개 메커니즘 metadata와 보호 후 target만. Q 원통계·환자별 자료·noise seed/실현 잡음은 넣지 않는다. 재현용 고정 seed 잡음은 public/구성 배열 검사에만 쓴다. 실제 target 생성은 운영체제 난수 기반의 독립 Gaussian 표본을 사용하고 seed를 출력하지 않는다.

외부 산출물의 ε는 고정된 공개 설정 하 Q→보호요약→합성의 한 실행에 관한 것이다. 이미 Q/V를 사용한 개발·선택 이력을 소급해 DP로 만들지 않는다. 모델/recipe의 데이터 의존 선택까지 포함한 전체 과정의 보호와는 구분한다. 유한정밀도 수치검사는 실수 Gaussian 정리의 별도 구현 보안 증명은 아니다.

## 이번 검산과 후속 첫 비교

- 기존 P/Q condition feature cache를 사용한 환자/class 집계가 두 encoder의 P/Q/pooled target에 일치하는가(기존 atol1e-12).
- 공개 C_c 선정 뒤 Q의 clipping-only 경로를 내부에서 검산한다. Q noise 생성·저장 및 효용 평가는 하지 않는다.
- 추가/제거·환자 교체·class 변경·반복 방문·없는 class·잘못된 분모·비유한 입력·count 안정화·직렬화 경계를 검사한다.
- Gaussian calibration을 별도 구현으로 확인하고, public/구성 배열로 noise→target→기존 objective 연결을 확인한다. 실제 encoder gradient 수리는 반복하지 않는다.

다음 실행을 제안한다: 기존 A2(P), 기존 A2(P+Q)를 재사용하고 seed101의 **clipping-only/no-noise A2 한 bank + 한 DP release의 A2 한 bank**를 추가한다. 동일128장·200회·microbatch16·초기 상태·최종PNG 규칙을 유지한다. 두 새 PNG를 고정한 뒤 동일 DenseNet/BioViL V 평가와 기존2000 환자 draw로 P→clipped→DP 차이를 보고한다. clipping-only는 비DP 내부 기준이며 보호 산출물이 아니다. 기존속도 기준 합성 약148분, 준비/평가 별도. 이 명세는 두 bank 실행 권한이 아니며 본 작업은 구현/검산에서 멈춘다. 한 noise 반복은 DP 평균 효용/분산의 증거가 아니다.

출처:
- [Balle–Wang, ICML2018, Theorem8](https://proceedings.mlr.press/v80/balle18a/balle18a.pdf)
- [Dwork–Roth, post-processing/composition](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [IBM GaussianAnalytic implementation](https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py)

