# DINO 단독 환자-DP 대조 결과

2026-09-24 KST. DINO 단독 DP의 DenseNet AUROC/AP는0.672935/0.080814로 기존 A2 DP 두 결과보다 점추정이 높았다. 두 A2−DINO 차이 구간은 모두0을 포함하므로 DINO의 우위를 확정하지 않는다. 이번 비교에서 A2 결합의 DP 아래 추가 우위는 확인되지 않았다. 결과에 따라 잡음을 다시 뽑지 않았다.

이번 범위는 DINO 전용 보호 경로 검산 → 독립 Q 요약1회 →128장·200회·seed101 bank1개 →최종PNG 고정 →기존 V 개발 비교까지 완료다. 연구 단계2는 in_progress이며 논문 전체·독립 수신자·강한 선행 비교 완료는 아니다.

## 결과

| 조건 | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP |
|---|---:|---:|---:|---:|
| 기존 A2 공개전용 | 0.542669 | 0.053838 | 0.714645 | 0.077790 |
| 기존 DINO 비DP | 0.635777 | 0.071016 | 0.565009 | 0.055268 |
| 새 DINO DP8 | 0.672935 | 0.080814 | 0.707482 | 0.085203 |
| 기존 A2 DP8 잡음1 | 0.658184 | 0.076739 | 0.730769 | 0.080103 |
| 기존 A2 DP8 잡음2 | 0.645283 | 0.074946 | 0.724648 | 0.076498 |

주 비교는 같은 개별 ε8/δ10⁻⁵의 A2와 DINO다. A2 공개전용은 맥락용이며 DINO 전용 공개 대조가 아니다. 기존 DINO 비DP와의 차이는 DINO의 clipping·목표 재구성·잡음을 합친 차이이며 잡음만의 인과효과로 분리하지 않는다.

| DenseNet 대응 비교 | AUROC 차이 [환자95% CI] | AP 차이 [환자95% CI] |
|---|---|---|
| A2 DP1 − DINO DP | -0.014751 [-0.054676, +0.027099] | -0.004075 [-0.022046, +0.011309] |
| A2 DP2 − DINO DP | -0.027652 [-0.079732, +0.028155] | -0.005868 [-0.026980, +0.013459] |

사전 기술적 판정: `MIXED_OR_NO_CLEAR_A2_ADVANTAGE`. 각 비교에서 ΔAUROC>0·환자CI하한>0·ΔAP≥0 여부를 함께 확인했다. 이 규칙을 결과 후 바꾸지 않았다. 2000회 동일 paired 환자 draw를 재사용했다. DINO는 새 잡음1회이고 A2는 기존 잡음2회다. 두 비교가 같은 DINO bank를 공유하므로 독립적인 두 번의 DINO 검증이 아니다. 환자 구간은 고정된 bank·실현 잡음에 조건부이며 합성 seed/DP 잡음 전체 변동을 포함하지 않는다. 여러 독립 비교를 끝낸 확증 검정이나 잡음 분포 전체의 우위로 해석하지 않는다.

DenseNet/V는 재사용된 개발 자원이다. BioViL은 A2 제작 source이며 새 DINO 제작에는 쓰이지 않았다. 마지막200회 PNG 후에만 평가했고 좋은 checkpoint나 좋은 A2 잡음 하나를 고르지 않았다.

## 공정한 단독 보호 경로

DINO만의 환자/class 방문 평균 gradient를 K4×34=136좌표로 만들었다. P-only norm q95로 class0 C=2.063911279562, class1 C=1.779795529652를 정했다. 양성 P 환자6명이라는 보정의 한계는 그대로다. Q/V를 보고 기준을 변경하지 않았다.

`0.5[clip(g0,C0)/C0; clip(g1,C1)/C1; b0; b1]`의274좌표 전체를 보호한다. A2의546좌표 보호값을 잘라 재사용하지 않았다. 한 환자 기여norm≤1, add/remove 감도1, analytic Gaussian σ=0.600229072259. 단독 DINO에도 전체 ε8 예산을 적용했다. A2와 똑같은 noisy class mass 안정화·sum 일관성 투영·P pooled 가중·0목표 public fallback을 사용했다.

새 방식이나 균형화 손실을 더하지 않았다. 비DP DINO의 full-weight K4 cosine·anchor/TV·optimizer·공개 초기화·200회 일정을 유지했다. 같은200회 비교이며 A2와 계산비용까지 같다는 뜻은 아니다.

## 검증과 실제 계산량

기존 DINO pooled target 재현 최대오차 5.551e-17, clipping을 끈 무잡음 재현 5.551e-17, 제한 후 독립 계산 5.551e-17. 공개 모의 target 직렬화/feature gradient 차이0. 환자 기여norm의 실제 최대 0.983596919223. 잘못된 입력5종 및 잘못된 job2종을 거부했다. 구성 경계·empty Q·음수 noisy 분모 검사도 통과했다.

내부 Q class clipping 환자 수는 [57, 16]이다. 이 수치는 비DP 내부 진단이며 보호 PNG 묶음에 포함하지 않았다. 이것만으로 성능 차이의 원인을 단정하지 않는다.

기존 DINO training/replay bytecode는 그대로다. 기존 A2 seed101과 renderer·optimizer·RNG·활성 pyramid 초기값이 정확히 같았다. 학습200회에서 frozen encoder/target 불변, 유한 loss/gradient, parameter 갱신, 최종PNG128장 검증을 통과했다. DINO 학습forward 204,800장, backward 102,400장. 검증/학습용으로 새 Q/V 픽셀이나 모델 특징을 추출하지 않았다.

- 합성·저장: **35.11분**.
- 평가: 11.89초. 새 PNG source/recipient 각128장, V 특징 재사용.
- 구현 착수부터 비교 완료: **43.66분**. 최초 예상50~70분, bank60분·전체90분 상한 안에 완료했다.
- GPU peak allocated 2646.22MiB / reserved 2892.00MiB.
- 완료시 남은 디스크 4.57GiB. 이번 bank의 중복 checkpoint만 정리했으며 과거 결과·cache·고정 코드 hash를 보존했다.

## 보호 회계와 범위

새 DINO_DP8_release01은 개별(8,10⁻⁵) Q 보호 요약1회다. 결과를 보고 다시 noise를 뽑지 않았다. 기존 A2 두 요약과 새 DINO를 모두 공개하면 기본 합성 보수적 상한은 **(24,3×10⁻⁵)**이다. 각 개별 실행의 ε를24로 바꿔 부르는 것이 아니며, A2 두 요약만의 기존 상한은16/2×10⁻⁵이다. [Dwork–Roth, Theorem3.16](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

동일 요약에서의 합성/PNG 후처리는 추가 private query를 만들지 않는다. 보호는 고정 공개recipe의 Q 요약에 한정한다. 과거 비DP Q/V 개발 선택, 내부 원자료 진단, 평가 보고서 전체를 소급해 DP라고 주장하지 않는다. OS Gaussian 구현의 수치 검산과 별도의 유한정밀도 보안 증명도 구별한다. 외부 공개/전송은 수행하지 않았다.

## 산출물과 종료

- [사전 계약](RECEIVER_DINO_PATIENT_DP_CONTRACT_20260924.md)
- [실행 결과](../../code_working/_reports/receiver_dino_dp8_20260924_v1/campaign_result.json)
- [평가·구간](../../code_working/_reports/receiver_dino_dp8_20260924_v1/evaluation/result.json)
- [보호 경로 검산](../../code_working/_reports/receiver_dino_dp8_20260924_v1/protection_verification.json)
- [공개용 파일목록](../../code_working/_reports/receiver_dino_dp8_20260924_v1/protected_png/release_manifest.json)
- [누적 회계](../../code_working/_reports/receiver_dino_dp8_20260924_v1/privacy_release_ledger.json)

허용된 한 bank 비교를 완료하고 정지했다. 추가 잡음·seed·예산 탐색·학습 연장·Expert/Reserved/확인용 receiver는 실행하지 않았다. 이 대조는 두 source의 필요성을 묻는 직접 제거실험이며 LGM/HMDC/Dosser 등 강한 선행 비교 전체를 대신하지 않는다.
