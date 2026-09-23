# DINO 단독 200회 대조 결과 — 2026-09-23

기록: 2026-09-23T05:01:38.170150+00:00

## 판단

DINO 단독보다 A2가 DenseNet AUROC를 더 높이면서 BioViL 판별력도 유지했다. DINO 단독의 DenseNet AUROC/AP는 0.635777/0.071016이고 A2는 0.671734/0.078565다. A2−DINO AUROC +0.035957의 동일 환자 bootstrap 95% 구간은 [+0.003202, +0.070914]로 양수다. BioViL AUROC는 DINO 단독 0.565009에서 A2 0.725230으로 +0.160221이며 구간은 [+0.102444, +0.218886]이다. 이번 고정된 recipe에서는 DINO만 쓰는 것으로 A2의 두 평가 결과를 설명할 수 없고, 두 source를 결합하는 실제 추가 가치가 관측됐다.

DINO 추가 자체도 중요한 역할을 했다: DINO−A1의 DenseNet AUROC는 +0.099320이다. 따라서 기존 A2−A1의 개선을 모두 결합의 독자적 효과라고 부르지는 않는다. A2−DINO의 DenseNet AP 차이 +0.007549는 구간 [-0.007642, +0.021304]이 0을 포함하므로 AP 우위까지 확정하지 않는다. A2는 합성 73.92분, DINO 단독은 34.71분으로 약2.13배 비용이다. 같은200회·한seed 개발 대조에서의 효용/비용 결과이며, 이점의 인과 기전·반복 재현·동일 계산비용 우위·DP 효용·최종 unseen 전이 또는 CVPR 기여를 입증한 것은 아니다.

## 동일 200회·seed101 결과

| 조건 | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP | 합성 분 |
|---|---:|---:|---:|---:|---:|
| BioViL 단독 A1 | 0.536456 | 0.047955 | 0.731632 | 0.079062 | 40.09 |
| BioViL+DINO A2 | 0.671734 | 0.078565 | 0.725230 | 0.076155 | 73.92 |
| DINO 단독 | 0.635777 | 0.071016 | 0.565009 | 0.055268 | 34.71 |

A1/A2의 결과·예측·bootstrap·비용은 재사용했다. 새 결과는 DINO 단독 128장 한 bank다. 같은 update 수이며 같은 연산량은 아니다. 합성 시간은 worker 시작·학습·체크포인트·PNG 저장을 포함한다.

## 환자 단위 차이

| 비교 | AUROC 차이 [환자 bootstrap 95%] | AP 차이 [환자 bootstrap 95%] |
|---|---:|---:|
| DenseNet_DINO minus DenseNet_A1 | +0.099320 [+0.035794, +0.160294] | +0.023061 [+0.008303, +0.045797] |
| DenseNet_A2 minus DenseNet_DINO | +0.035957 [+0.003202, +0.070914] | +0.007549 [-0.007642, +0.021304] |
| BioViL_DINO minus BioViL_A1 | -0.166623 [-0.227699, -0.107304] | -0.023793 [-0.039176, -0.006921] |
| BioViL_A2 minus BioViL_DINO | +0.160221 [+0.102444, +0.218886] | +0.020887 [+0.004736, +0.035155] |
| DenseNet_DINO minus DenseNet_B500 | +0.067802 [+0.022140, +0.112942] | +0.010918 [-0.005371, +0.029663] |
| DenseNet_DINO minus DenseNet_P_real | +0.017768 [-0.035016, +0.066381] | +0.006075 [-0.011107, +0.027372] |

V 2,026명/5,047장, 양성 223장. 기존 2,000회 paired 환자 bootstrap을 재사용했다. 구간은 seed101 bank에 조건부이며 합성 seed 불확실성은 포함하지 않는다. DenseNet과 V는 재사용된 개발 자원이다. 유의하지 않은 차이를 동등성 입증으로 해석하지 않는다. 마지막 B500/공개 실자료 비교는 참고이며 사적 추가 효용이나 동일 비용의 방법 우위를 분리하지 못한다.

## 고정한 방법

A2의 DINO checkpoint·P-only PCA16/q95·K4 head/변환·class별 환자 균등 P/Q 목표를 재사용했다. BioViL 손실을 빼고 DINO 조건 평균의 가중치1, anchor0.01/TV0.0001을 한 번 적용했다. 새 목표 추출은 없다.

A2와 같은 공개 template·labels64/64, AdamW lr0.01/decay0/betas(.9,.999)/eps1e-8, FP32 encoder·FP64 작은 CE gradient, microbatch16과 활성1/33/65/97/129/161을 유지했다. 마지막200회 PNG만 평가했으며 수신자 결과로 설정이나 checkpoint를 고르지 않았다.

## 검증

- 공개 P4 / K4 / microbatch4의 실제 gradient 대조: loss 차이 0.000e+00, 최대 gradient 차이 7.451e-09, relative L2 5.352e-08. 기존 허용기준 유지.
- A2 DINO 목표/head/조건 정확 일치, 단독 loss 가중치1, prior1회 확인.
- A2 step0과 renderer·optimizer·난수·활성 및 학습가능 상태 정확 일치.
- 200회 update 연속성, 유한 loss/gradient/parameter, 고정 encoder parameter/buffer·목표, 최종PNG 픽셀/크기/label/hash 통과.
- 기존 학습/평가 코드 불변. 이전 실행 이후 test_target_runtime.py에 추가된 선택적 handoff 누락 skip2줄은 diff·해시를 기록하고 보존했다. 최초 broad hash 검사는 GPU 착수 전에 중단됐다.
- 새 bootstrap sklearn 독립 계산 최대 오차 1.110e-16.
- 초기화 P 픽셀 접근 64개. Q/V 실영상 재추출 없이 새PNG의 BioViL/DenseNet 특징만 계산했다.

## 실측 비용과 저장공간

- 공개 연결 검사 7.92초.
- 정식 합성 34.71분 / 상한60분.
- 평가 10.80초.
- 계약 생성부터 최종 평가까지 35.36분. 코드 연결·착수전 검토는 그 앞에 별도 수행했다.
- 주학습 peak allocated 2.584GiB / reserved 2.824GiB.
- 새 encoder 연산: {"DINOv2": {"forward": 204848, "backward": 102432}, "BioViL": {"forward": 128, "backward": 0}, "DenseNet": {"forward": 128, "backward": 0}}. 정식 optimizer200회, 검사0회.
- 이번 bank 중복 checkpoint 523.67MiB 정리. step0, 마지막두/final, 모든 receipt·trace 보존.
- 현재 작업 산출물 385.57MiB, C 여유 9.17GiB.

## 증거 및 범위

code_working/_reports/receiver_dino_only_s200_20260923_v1/ 아래 campaign_contract.json, execution_bindings.json, changed_path_verification/result.json, DINO_result.json, bank_seal.json, evaluation/result.json, predictions_private.npz, paired_bootstrap.npz, campaign_result.json을 보존했다. 최종PNG는 banks/dev_DINO_condk4_101_s200/artifact/에 있다.

합성자료·목표는 NONDP_INTERNAL_Q_DERIVED다. 추가 seed·500회 연장·DP·Expert·Reserved·final receiver는 실행하지 않았다. 환자-DP·독립 unseen receiver·사적 추가 가치·논문 기여가 확인됐다는 결론은 내리지 않는다.

이번 DINO 단독 대조를 완료하고 멈춘다. 후속 후보는 고정된 A2와 단독 source 대조의 추가 seed 재현 확인이며, 이번 범위에서 실행하지 않았다. 강한 선행·Q 추가효용·환자-DP·독립 receiver 주장은 별도 근거가 필요하다.
