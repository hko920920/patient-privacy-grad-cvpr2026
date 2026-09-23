# A2 공개전용 대조 결과 — 2026-09-23

기록: 2026-09-23T08:38:03.947865+00:00

## 판단

동일 A2·seed101·공개 초기상태·200회 일정에서 Q를 추가한 개발 효용이 관측됐다. DenseNet AUROC는 공개 P-only 0.542669에서 P+Q 0.671734로 +0.129065이며, 고정된 두 bank에 조건부인 환자 bootstrap 95% 구간은 [+0.088112,+0.174088]이다. AP도 0.053838에서 0.078565로 +0.024728, 구간 [+0.008428,+0.041644]이다. 따라서 이번 고정 조건에서는 두 source의 결합만으로 같은 성능이 나온 것이 아니라 Q의 정보를 추가한 목표가 개발 receiver에서 추가 가치를 제공했다. 앞선 두 seed의 결합 이득과 이번 자료 범위 대조는 서로 다른 질문에 답한다.

사적 추가 효용 대조는 seed101 한 번이며, 앞선 DINO 대 A2 두 seed 반복이 P-only 대 P+Q의 반복까지 대신하지 않는다. DenseNet/V는 이미 사용해 온 개발 자원이고 구간은 합성 seed 불확실성을 포함하지 않는다. BioViL AUROC 차이 +0.010585의 구간 [-0.010778,+0.031693]과 AP 차이 -0.001635의 구간 [-0.014650,+0.006235]는 모두0을 포함한다. BioViL 개선까지 확정하거나 독립 수신자 일반화를 주장하지 않는다. 공개 P의 class-present 양성 환자는6명, Q는114명이며 이번 비교는 현재 P에 Q를 추가한 가치다. 더 큰/다른 공개자료 대비 사적 정보의 필수성이나 DP 후 효용을 입증한 것은 아니다. 환자-DP·강한 선행 대비 차별성·독립 확인은 아직 수행하지 않았다.

## 같은 seed101의 자료 범위 비교

| 평가 모델 | A2(P) AUROC | A2(P+Q) AUROC | 차이 | A2(P) AP | A2(P+Q) AP | 차이 |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet | 0.542669 | 0.671734 | +0.129065 | 0.053838 | 0.078565 | +0.024728 |
| BioViL | 0.714645 | 0.725230 | +0.010585 | 0.077790 | 0.076155 | -0.001635 |

차이는 기존 A2(P+Q)−새 A2(P)다. 기존 P+Q 최종 bank/예측/환자 bootstrap은 재사용했고, 새로 만든 것은 공개 P 목표의128장·200회 bank 하나다.

- DenseNet AUROC 차이 95% 구간: [+0.088112, +0.174088].
- DenseNet AP 차이 95% 구간: [+0.008428, +0.041644].
- BioViL AUROC 차이 95% 구간: [-0.010778, +0.031693].
- BioViL AP 차이 95% 구간: [-0.014650, +0.006235].

구간은 고정된 두 bank를 조건으로 동일 V 환자를 대응 재표집한2,000회 percentile95 결과다. 합성 seed 불확실성을 포함하지 않는다. V는 2026명/5047장, 양성223장이다. DenseNet과 V는 개발 자원이고 BioViL은 제작 source다.

## 목표 선택과 일치성

- P813장/672명, class-present 환자 수 음성669/양성6. 방문 평균→환자/class 평균→class0.5 규칙을 그대로 사용했다.
- 기존 두 encoder 파일의 P_sums/P_counts/P_gradient만 복사했다. 각각 P 분모로 독립 재계산한 최대오차 0.000e+00; 새 파일에는 Q/pooled block이 없다.
- 기존 P 목적과 새 파일을 읽은 목적의 digest가 정확히 같다. 새 파일에서 pooled를 선택하는 요청은 거부됐다.
- K4 조건/head·전처리·공개 projection·모델·optimizer·128장·200회·seed101·microbatch16·해상도 일정을 유지했다. anchor0.01/TV0.0001과 source equal mean도 같다.
- 기존 A2의 공개 template·최초 renderer/optimizer/RNG/활성상태가 학습 시작 전에 정확히 일치했다. 저장된 step0을 비교용으로 읽었으며 기존 학습 결과를 이어 학습하지 않았다.
- 새 wrapper는 P-only job 검증을 연결하며 실제 학습·저장·재개·PNG 연산은 검증된 repeat_runtime._run을 직접 재사용했다. 기존 source 파일은 변경하지 않았다.
- 최종200 PNG 저장 및 pixel/label/hash 검증 후에만 개발 평가했다. 두 encoder/target 불변,200회 update 연속성·유한 gradient/parameter 검증이 통과했다.

## 실행량과 비용

- 목표 준비/선택 검산 3.22초. 새 P/Q 특징 추출0, 검산 encoder F/B0, 검산 optimizer0.
- 합성·저장 74.35분. 기존 A2(P+Q)는 73.92분. 합성 worker 상한120분을 유지했다.
- 평가 11.12초; 실행 계약 기록부터 완료까지 74.77분. 코드 연결 시간은 별도다.
- 학습 encoder별204,800 image forward/102,400 backward, 정식 optimizer200회. 평가 PNG forward는 BioViL128/DenseNet128이며 V·Q 원영상 forward0.
- peak allocated 5.921GiB. 중복 checkpoint 523.67MiB 정리, 초기·마지막두/최종·모든 receipt/trace 보존. 디스크 여유 7.47GiB.
- bootstrap의 독립 sklearn 검산 최대오차 1.110e-16.

## 산출물 및 완료 범위

code_working/_reports/receiver_a2_public_s200_20260923_v1/:
preparation_contract.json, target_verification.json, targets/, reference_initialization.json,
campaign_contract.json, job.json, execution_bindings.json, A2_P_result.json, bank_seal.json,
evaluation/result.json, evaluation/predictions_private.npz, evaluation/paired_bootstrap.npz,
campaign_result.json, banks/dev_A2_P_condk4_101_s200/artifact/.

새 합성 목표는 공개 P만 사용했다. 비교에는 비DP Q 유래 기존 A2와 개발 V 결과가 포함되므로 전체 비교 산출물을 private 내부 기록으로 보존한다. DP·Expert·Reserved·확인용 수신자·추가 seed·학습 연장은 실행하지 않았다.

이번에 허용된 공개전용 A2 한 bank의 준비·학습·저장·개발 평가와 기록을 완료했다. 두 source 결합의 반복 이득에 이어 현재 P/Q 설정에서 Q의 추가 효용 근거를 확보했다. 다음 연구 판단은 이 추가 효용을 환자 수준 DP 아래 유지할 방법과 비교 범위를 구체화하는 것이며, 이번 결과만으로 DP·추가 seed·학습 연장·Expert·Reserved·확인용 receiver를 자동 실행하지 않는다.
