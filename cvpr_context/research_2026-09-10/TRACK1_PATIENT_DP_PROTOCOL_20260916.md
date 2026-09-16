# 방향1 환자DP 첫 비교 — 공개 보정과 고정 실행 범위

2026-09-16. [비DP 첫 결과](TRACK1_CAPACITY_RESULTS_20260916.md)의 작은 잔차층 적응이 **환자DP 아래 유지되고, 같은 보정층의 반복 학습 대비 의미 있는 효용·비용 선택지를 제공하는가**를 확인한다. 이 문서는 비교 결과 전에 작성한 범위다. 공개 선택값과 결과는 별도 아티팩트에 저장한다.

## 논리와 이득·대가

고정 특징·이차손실에서 A/B는 학습에 필요한 W 의존 정보를 보존한다. 통계를 한 번 보호한 뒤 해를 구하는 구조가 반복 gradient 보호와 다르다. 반면 고차원 Gram 잡음은 역행렬에서 증폭될 수 있고, PSD floor는 분산을 줄이는 대신 추가 편향을 만든다. 같은 head의 DP-SGD도 backbone 역전파가 없으므로 공통 비용을 동일하게 계산한다.

static16이 이전 비DP 개선의86.15%를 얻었으므로 full64와 함께 비교한다. 공개 보정자료 자체의 적응을 사적 학습의 이득으로 잘못 세지 않도록 **공개32명만으로 학습한 비DP head**를 추가한다.

## 공개 자료

기존 quality32명×2장=64장은 기존 fit/selection/calibration/test400명과 환자가 겹치지 않는다. 이전 모델·특징·projection·basis를 유지해 영상당8개 시점/noise, 총512기록을 추출했다. GPU513F(512+warmup),0B,47.386초. 통계2.359초,독립검산9.102초·15,202검사PASS다. 이후 이32명은 독립 품질test가 아니다.

[공개 계약](../../code_working/_reports/frozen_residual_public32_20260916_v1/contract.json) · [검산](../../code_working/_reports/frozen_residual_public32_20260916_v1/independent_verification.json) · [추출 코드](../../code_working/frozen_residual_head/run_public.py).

## 공개 선택 규칙

공개32명을 고정 두fold16/16으로 나누어 양방향 검증한다. 각train16명을5회 반복한80개 공개slot은 N0=80의 잡음·분모 척도를 맞추기 위한 대리자료다. 80명의 독립 환자가 아니다. 보호학습80명·개발평가40명으로 C·학습률·floor를 선택하지 않는다.

| 항목 | 결과 전 정한 작은 후보 범위 |
|---|---|
| SSP 척도 | 공개train A/B의 Frobenius norm 중앙값 a0/b0 |
| SSP 6후보 | joint norm C의 분위수0.5/0.9 × floor배율0/1/2 |
| floor 척도 | nu_A=(sigma*C*a0/80)*sqrt(2*d). 확률보장이 아닌 공개 잡음척도 |
| SGD 6후보 | 공개ridge해와0 사이 고정4점의 gradient norm C 분위수0.5/0.9 × 초기학습률0.25/L,0.5/L,1/L |
| L | 공개평균Gram의2*(최대고유값+lambda) |
| schedule | 처음80%step 일정, 마지막20%에 초기학습률의1%까지 선형감소 |
| sampling/초기화 | 환자Poisson q=.1,분모q*N0,0초기화 |
| T 후보 | 500/2000/8000 |
| T 충분성 | 실제같은Poisson optimizer,noise0/clip해제. 각fold의3고정seed 평균이 초기ridge objective gap의1% 이내인 첫T |
| 후보 선택 | 통과한T에서 각방법6후보×2fold×3고정noise반복의 평균validation MSE. 최선seed 선택 없음 |

충분성을 통과하지 않으면 baseline을 정상이라고 부르거나 사적 비교를 강행하지 않는다. 선택한 규칙으로 공개32명 전체에서 척도를 다시 계산한다. λ=.001은 유지하지만 양의floor는 추가 스펙트럼 정규화라고 명시한다. 실제T의 Opacus/Google RDP 중 큰ε가8이하가 되게σ를 맞춘다. δ=1e-5,환자추가/제거 인접성,공개N0=80이다.

## 본 비교와 대조

full64/static16 × joint-SSP/같은headDP-SGD 네 칸을 학습80명에서 실행하고 별도개발40명에서 평가한다. 같은환자당4장·8draw·finite목적·캐시를 준다. 새UNet계산은0이다.

각칸의 사전고정16개noise반복을 전부 저장한다. 공개선택과 다른 sampling/noise stream을 사용한다. 환자별MSE 및16회 평균·표준편차·범위를 제시하며 이를16개의독립데이터셋으로 부르지 않는다.

대조는 기반모델,공개32명비DP회귀,학습80명비DP회귀,SSPnoise0/floor0,SSPnoise0/선택floor,SGDnoise0/clipping유지·해제각3회다. 표현·공개적응·clipping·추가정규화·잡음·최적화부족을 구별한다. private성능을 본 뒤 C·λ·seed를 다시 고르는 것은 이번 계약에 없다.

joint환자벡터의 민감도는C/N0다. SGD는 매step 환자gradient clip 후 Gaussian 잡음을 더하고 q*N0로 나눈다. 빈batch에도 noise와ridge update를 수행한다. 공개ridge gradient 2*lambda*W는 환자clipping밖에서 한 번 더한다. 같은finite목적의환자gradient 2*(A_uW-B_u)를 두 방법 모두 내부캐시에서 계산한다.

## 검산과 해석 범위

[보호 연산 구현](../../code_working/frozen_residual_head/dp_mechanisms.py)은 원래비DP소스와 분리했다. 별도검산기는 생산자수치함수를import하지 않고 SSP 재구성,SGD mask/noise/weight 경로,전체평가MSE·요약·공개선택/회계/출처를 대조한다. 동결시점과 실제검사범위를 그대로 기록한다.

NIH공개자료에서의 보호메커니즘 모사다. 개별고정메커니즘모델의 DP회계와 raw통계·비DP대조·seed를 포함한 내부연구폴더전체의보호를 구별한다. 실제여러출력을공개할때는 합성회계가 필요하며 이전비DP결과를 소급해보호하지 않는다.

denoising개선은 실제생성품질·임상효용·최강선행우위·신규성·채택확인과 구별한다. 특정초기설정의손실도 연구방향전체의불가능으로 확대하지 않는다. [선행과 다음 비교 명세](TRACK1_PATIENT_DP_COMPARISON_PLAN_20260916.md)의 목적을 유지한다.

## 공개 선택 완료 기록 — 본 평가 전

공개 선택은39.577초에 완료했다. 두head 모두500step에서는 수렴 기준 미달,2000step에서 통과했다. full의두fold평균 normalized gap은0.003729/0.004268,static은0.000294/0.000282였다. 각방법6후보×2fold×3seed=144개후보fit와24개수렴대조를 전부 보존했다.

두head의SSP는clip분위수0.5·floor배율1,SGD는clip분위수0.5·LR배율0.5가 선택됐다. SSP sigma=.6376701852,SGD q=.1/T2000 sigma=2.9650482183이며 두RDP구현의큰epsilon은8이하다.

공개CV에서 선택된full/static SSP의MSE는.178652/.178046,SGD는.174149/.174650이었다. 이는공개대리자료결과이고 본평가40명결과가 아니다. 이차이를숨기거나평가자료로후보를다시고르지않는다. 비DP공간·시간보정층의적응,DP유지,일회통계보호우위는서로다른주장이다.

자료로딩에서 공개split이름을public32로잘못검사한오류를첫profile에서발견했다. 후보성능계산전에실제public schema로수정했다. 이전코드·계약·실패protocol과전후hash는공개calibration폴더의schema_correction_01에보존했다. 수치·선택규칙은변경하지않았다.

[선택 계약](../../code_working/_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1/selection_contract.json) · [전체 후보](../../code_working/_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1/candidate_summaries.json) · [선택값](../../code_working/_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1/params.json) · [회계](../../code_working/_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1/accounting.json).

공개정보를더강하게활용하는표준대안은[공개중심·공개Gram·한번의보호갱신수식검토](spec_sources/track1_public_reference_alternatives_20260916.md)에별도로남겼다. 현재네칸을바꾸지않으며,그대안의효용이나새기여가검증됐다는뜻이아니다. 현재SGD의반복수는noise없는수렴검사에서고른것이므로DPnoise까지고려한전역최적반복수를찾았다고주장하지않는다.
