# A2 공개전용 seed202 및 두 초기화의 Q 추가 효과 — 2026-09-23

기록: 2026-09-23T10:07:29.439689+00:00

## 판단

같은 A2 구성에서 Q를 추가한 DenseNet 개발 AUROC 이득이 두 합성 초기화에서 반복됐다. seed101은 P 0.542669 → P+Q 0.671734 (Δ+0.129065), seed202는 P 0.563388 → P+Q 0.670355 (Δ+0.106967)이다. 각 고정 bank 쌍의 환자 bootstrap AUROC 차이95% 구간은 모두 양수다. 두 seed 평균 차이는 +0.118016이다. DenseNet AP 차이는 +0.024728/+0.021118로 모두 양수지만, seed202 AP 차이95% 구간[-0.001204,+0.041792]은0을 포함하므로 AP까지 반복적으로 유의한 우위라고 결론내리지 않는다.

이번 결과는 비DP·고정된 K4 조건/head·두 합성 초기화·재사용된 개발 DenseNet/V에서 얻은 Q 추가 가치다. 환자 bootstrap은 고정 bank에서의 환자 표집 불확실성이며 합성 seed 모집단/DP 잡음/독립 수신자 일반화를 포함하지 않는다. BioViL AUROC 변화는 두 seed 모두 작고 구간이0을 포함한다. 공개 P의 양성 class-present 환자는6명이며, 이 결과를 모든 공개자료 대안에 비해 Q가 필수라는 주장으로 확대하지 않는다. 환자-DP 효용·동일 비용 우위·강한 선행 대비 가치·독립 확인은 아직 검증하지 않았다.

## 결과

| 평가 모델 | seed | P AUROC | P+Q AUROC | Q 추가 차이 | P AP | P+Q AP | Q 추가 차이 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet | 101 | 0.542669 | 0.671734 | +0.129065 | 0.053838 | 0.078565 | +0.024728 |
| DenseNet | 202 | 0.563388 | 0.670355 | +0.106967 | 0.058904 | 0.080022 | +0.021118 |
| BioViL | 101 | 0.714645 | 0.725230 | +0.010585 | 0.077790 | 0.076155 | -0.001635 |
| BioViL | 202 | 0.716413 | 0.728906 | +0.012493 | 0.075553 | 0.082269 | +0.006716 |

차이는 모두 A2(P+Q)−A2(P)이다. seed101의 두 bank와 seed202의 P+Q bank는 기존 결과를 재사용했다. 새로 학습한 것은 P-only seed202 한 bank뿐이다.

- DenseNet, seed101, AUROC 차이95% 구간: [+0.088112, +0.174088].
- DenseNet, seed101, AP 차이95% 구간: [+0.008428, +0.041644].
- DenseNet, seed202, AUROC 차이95% 구간: [+0.053575, +0.161466].
- DenseNet, seed202, AP 차이95% 구간: [-0.001204, +0.041792].
- BioViL, seed101, AUROC 차이95% 구간: [-0.010778, +0.031693].
- BioViL, seed101, AP 차이95% 구간: [-0.014650, +0.006235].
- BioViL, seed202, AUROC 차이95% 구간: [-0.013256, +0.036666].
- BioViL, seed202, AP 차이95% 구간: [-0.004827, +0.016325].

두 seed의 차이를 단순 평균한 DenseNet AUROC 차이는 +0.118016, AP 차이는 +0.022923이다. 이는 예측 앙상블 성능이나 합성 초기화 모집단의 신뢰구간이 아니다.

각 구간은 고정된 bank 쌍을 조건으로 같은 V 환자를 재표집한2,000회 percentile95 결과다. 합성 초기화·조건/head 생성의 불확실성을 포함하지 않는다. 두 초기화는 기존 K4 조건/head·목표를 고정하고 공개 template 선택과 합성 초기화/학습 난수만 바꾼 반복이다.

V는2026명/5047장/양성223장이다. DenseNet과 V는 반복 사용된 개발 자원이며 BioViL은 학습에 사용한 source이다. 독립 수신자/최종 평가 결과로 부르지 않는다.

## 변경 범위와 검증

- 기존 P 전용 목표 파일과 handoff를 그대로 사용했다. P813장/672명, class-present 환자 수 음성669/양성6. 환자/class 내부 방문 평균 → 해당 class 환자 균등 평균 → class별0.5 가중을 유지했다.
- 두 encoder의 목표 파일 hash와 objective digest가 seed101 P 전용 결과와 정확히 같다. 새 P/Q 특징 추출0, 새로운 목표 계산0, 준비 단계 pixel/model forward/backward/update0.
- 합성/template seed202, 조건/head seed101을 구분했다. BioViL+DINOv2, 공개 projection,128장·200회·microbatch16, optimizer·해상도 활성 일정·anchor/TV 가중치는 그대로다.
- 기존 PQ202와 template tensor, 초기 renderer/optimizer/RNG/활성 상태가 첫 update 전에 정확히 일치했다. 기존 checkpoint를 이어 학습하지 않았다.
- 검증된 repeat_runtime._run을 직접 재사용했다. 이전 실행기의 gradient/재개/PNG 검증을 다시 실행하지 않았다.
- 새 코드의 변경 범위는 seed202 P-only job 검증, 기존 목표/초기 상태 연결, 두 seed 평가 집계다. 이전 runtime 및 모든 기존 결과 파일은 보존되고 hash가 유지됐다.
- 200회 연속 update, 유한 loss/gradient/parameter, 고정 encoder/target 불변, 최종128 PNG pixel/label/hash 검증이 통과했다.
- 최종PNG 고정 뒤에만 평가했다. 기존 V 특징/기존 bank 예측/환자 bootstrap 배열을 재사용했다. 새 PNG forward는 BioViL128/DenseNet128이며 Q/V 원영상 forward0.
- bootstrap 독립 sklearn 검산 최대 오차1.110e-16. 결과를 본 뒤 계수·checkpoint·학습량을 선택하지 않았다.

## 실제 비용과 보존

- 목표 재사용/초기 상태 결속 준비: 2.06초.
- 새 P-only seed202 합성·저장: 73.91분. 기존 P-only seed101은74.35분, PQ202는74.56분이다.
- 새 PNG 평가 및 두 seed 집계: 10.47초. 실행 계약 생성부터 완료까지74.32분(앞선 코드 연결 시간 별도).
- 학습 encoder별204,800 image forward/102,400 backward, optimizer200회. peak allocated5.921GiB.
- 이 bank의 중복 중간 checkpoint523.67MiB를 기존 규칙대로 정리했다. 초기 checkpoint·마지막두/최종 checkpoint·모든 receipt/trace/PNG는 보존했다. 디스크 여유7.08GiB.
- cumulative worker 상한120분 내 완료했고 추가 bank·학습 연장·DP·Expert·Reserved·확인용 수신자를 실행하지 않았다.

## 산출물

code_working/_reports/receiver_a2_public202_s200_20260923_v1/:
campaign_contract.json, job.json, target_verification.json, execution_bindings.json,
A2_P_result.json, bank_seal.json, evaluation/result.json, evaluation/predictions_private.npz,
evaluation/paired_bootstrap.npz, campaign_result.json, banks/dev_A2_P_condk4_202_s200/artifact/.

이번 학습의 target은 공개 P만 사용했다. 비교에는 비DP Q 유래의 기존 결과와 개발 V 예측이 포함되므로 전체 비교 산출물은 내부 기록으로 유지한다. 과거 Q 접근이나 이번 개발 선택을 DP로 소급하지 않는다.

다음 연구 질문은 사전 고정한 작은 환자-DP 비교에서 A2(P+Q)가 공개전용 A2(P)보다 추가 효용을 유지하는가이다. 이번에는 DP 설계/실행을 시작하지 않았다. 보호 단위·class 분모/가중·결합 query·clipping/잡음 및 비교 범위를 구체화한 다음 별도 작업으로 진행한다. 추가 seed·학습 연장·Expert·Reserved·확인용 수신자는 자동 실행하지 않는다.
