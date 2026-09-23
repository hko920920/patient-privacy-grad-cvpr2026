# DINO/A2 seed202 반복 결과 — 2026-09-23

기록: 2026-09-23T07:06:14.487543+00:00

## 판단

새 합성 초기화 seed202에서도 A2(BioViL+DINO)의 결합 이득이 관측됐다. DenseNet AUROC는 DINO 0.597164에서 A2 0.670355로 +0.073192이며, 두 고정 bank에 조건부인 환자 bootstrap 95% 구간은 [+0.024720,+0.121602]다. AP도 0.060585에서 0.080022로 +0.019437, 구간 [+0.004241,+0.034900]이다. seed101과 seed202 모두 A2의 DenseNet AUROC/AP 점추정이 DINO 단독보다 높았다. 두 초기화에서 A2 AUROC는 0.671734/0.670355였고, DINO는 0.635777/0.597164였다. 이는 강한 단독 source를 넘는 결합의 개발 효용이 한 초기화에만 국한되지 않았다는 근거다.

이번 seed202에서 BioViL AUROC도 DINO 0.632435 대비 A2 0.728906으로 높았다. BioViL은 A2 제작에 사용한 source이며, DenseNet/V는 이미 설계 선택에 사용한 개발 자원이다. seed101의 DenseNet AP 차이 구간은 여전히 0을 포함한다. 조건/head/변환과 목표는 고정했고 공개 template·초기 residual/학습 RNG를 바꾼 반복이다. 두 seed의 양성 방향만으로 seed 모집단의 안정성이나 독립 unseen 전이를 확정하지 않는다. A2는 두 source 계산을 사용해 seed202 합성 74.56분, DINO는 34.43분으로 약 2.17배 비용이다. 같은 업데이트 수에서의 효용 근거이며 같은 계산비용에서의 우위, 사적 Q의 순수 추가 효용, 환자-DP, 강한 선행 대비 차별성은 아직 확인하지 않았다.

## 두 합성 초기화의 결과

| Seed | Source | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP | 합성 분 |
|---:|---|---:|---:|---:|---:|---:|
| 101 | DINO | 0.635777 | 0.071016 | 0.565009 | 0.055268 | 34.71 |
| 101 | A2 | 0.671734 | 0.078565 | 0.725230 | 0.076155 | 73.92 |
| 202 | DINO | 0.597164 | 0.060585 | 0.632435 | 0.063448 | 34.43 |
| 202 | A2 | 0.670355 | 0.080022 | 0.728906 | 0.082269 | 74.56 |

101 결과는 기존 고정 bank의 예측·bootstrap을 재사용했다. 이번에 새로 실행한 것은 seed202 DINO/A2의128장·200회 두 bank다. 같은 update 수이며 같은 연산량은 아니다. 합성 비용은 worker의 시작·저장·PNG까지 포함한다.

## 대응 차이와 반복 해석

| Seed / 평가 | A2−DINO AUROC [95%] | A2−DINO AP [95%] |
|---|---:|---:|
| 101 / DenseNet | +0.035957 [+0.003202, +0.070914] | +0.007549 [-0.007642, +0.021304] |
| 101 / BioViL | +0.160221 [+0.102444, +0.218886] | +0.020887 [+0.004736, +0.035155] |
| 202 / DenseNet | +0.073192 [+0.024720, +0.121602] | +0.019437 [+0.004241, +0.034900] |
| 202 / BioViL | +0.096471 [+0.043158, +0.150194] | +0.018822 [+0.000260, +0.035330] |

DenseNet 두 seed 차이의 단순 평균: AUROC +0.054574, AP +0.013493.
BioViL: AUROC +0.128346, AP +0.019854.

이 평균은 개별 bank 지표 차이의 산술 평균이며 예측 ensemble 성능이 아니다. 두 seed만으로 합성 seed 모집단의 유의성이나 안정성을 확정하지 않는다. 표의 구간은 각 두 bank를 고정한 V 환자 표집의 2,000회 paired bootstrap 95% 구간이다. 합성 seed 불확실성은 포함하지 않는다.

V 2,026명/5,047장·양성223장, 기존 BioViL/DenseNet 특징과 환자 draw를 그대로 썼다. DenseNet과 V는 적응적으로 사용된 개발 자원이며 BioViL은 A2의 source다. 독립 unseen receiver 결과가 아니다.

## 고정한 것과 달라진 것

- target condition/head seed101, 모델 checkpoint·P-only PCA16/q95·K4 변환/head·class별 환자 균등 P/Q 목표는 그대로다. 신규 추출0.
- 공개 template 선택 및 초기 residual/학습 RNG만 seed202로 바뀌었다. 두 방법은 동일 seed202 template·renderer·optimizer·RNG·활성상태에서 출발했다.
- DINO loss1 대 A2의 BioViL0.5+DINO0.5, K4 평균, anchor0.01/TV0.0001을 유지했다.
- 128장·labels64/64·200회·microbatch16, AdamW lr0.01/decay0/betas(.9,.999)/eps1e-8, pyramid 활성1/33/65/97/129/161을 유지했다.
- 기존 seed101 코드를 수정하지 않고 별도 실행 인자 --seed202를 연결했다. 조건 seed와 합성 seed를 별도 검증한다.
- 두 최종PNG가 모두 고정된 뒤 평가했다. 학습 중 성능 조회, checkpoint/계수 선택, 추가 profile은 없다.

## 검사와 실측

- 명시적 seed 연결 검사: PASS_EXPLICIT_SYNTHESIS_SEED_AND_TARGET_SEPARATION; 2.828초, 환자 픽셀/encoder F·B/optimizer0.
- seed101 초기화·모든 activation 출력은 기존 경로와 정확 일치. seed202 residual은 변경되고 목표 digest는 그대로다.
- 실제 두 bank 초기 renderer/optimizer/RNG/활성상태 해시 정확 일치, 잘못된 seed의 checkpoint 재개 거부.
- 기존 gradient·저장/재개·PNG 검증을 재사용. 실제400회 update 연속성 및 유한 loss/gradient/parameter, 고정 encoder/target, 최종PNG pixel/label/hash 검사 통과.
- 새 bootstrap 독립 sklearn 검산 최대 오차 1.110e-16.
- DINO seed202 34.43분 / 상한60분.
- A2 seed202 74.56분 / 상한120분.
- 평가 10.34초, 계약 기록부터 완료까지 109.53분. 코드 연결은 계약 기록 전에 별도로 수행했다.
- peak allocated: DINO 2.584GiB, A2 5.921GiB.
- 새 encoder 연산: {"BioViL": {"forward": 205056, "backward": 102400}, "DINOv2": {"forward": 409600, "backward": 204800}, "DenseNet": {"forward": 256, "backward": 0}}. 정식 optimizer400회.
- 중복 checkpoint 1047.34MiB 정리. 각 bank step0·마지막두/final·모든 receipt/trace 보존. 디스크 여유 7.83GiB.

## 산출물·범위

code_working/_reports/receiver_repeat202_s200_20260923_v1/:
campaign_contract.json, seed_verification/result.json, execution_bindings.json,
DINO_result.json, A2_result.json, two_bank_seal.json, evaluation/result.json,
evaluation/predictions_private.npz, evaluation/paired_bootstrap.npz, campaign_result.json.
최종영상은 banks/dev_DINO_condk4_202_s200/artifact 및 banks/dev_A2_condk4_202_s200/artifact에 있다.

기존 seed101 실험·코드·목표·결과는 보존했다. 이번도 Q 유래 비DP 내부 개발이다. 추가seed·500연장·DP·Expert·Reserved·final receiver는 실행하지 않았다.

허용된 seed202 DINO/A2 두 bank의 학습·저장·개발 평가와 기록을 완료했다. 이번 양성 결과를 보존하며 추가 seed·학습 연장·새 loss·DP·Expert·Reserved·final receiver로 자동 확대하지 않는다. 다음 연구 판단에서는 확인된 결합 이득과 아직 미확인인 사적 추가 효용·보호 후 효용을 구분한다.
