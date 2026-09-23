# DINO 환자-DP 독립 잡음2 반복 결과

2026-09-24 KST. DINO 잡음2의 DenseNet AUROC는 잡음1보다 0.048807 낮아져, 이번에는 A2 두 bank보다 낮았다. 두 잡음의 AUROC 기술평균은 DINO 0.648531, A2 0.651733다. DINO의 점추정 우위는 반복되지 않았다. 네 A2−DINO 교차 비교의 AUROC/AP 환자 구간은 모두 0을 포함한다. 따라서 DINO가 항상 낫다는 결론도, A2의 일반적 우위를 확보했다는 결론도 지지하지 않는다. 현재 DINO를 강한 직접 대조로 유지하며, 두 잡음만으로 안정성 차이를 확정하지 않는다.

새 DINO DP2의 DenseNet AUROC/AP는 **0.624128/0.067322**다. 기존 DP1은 0.672935/0.080814이며, 둘 다 공개 clipping·274좌표 query·ε8/δ10⁻⁵·조건/head·합성 seed101을 유지하고 DP 잡음만 독립적으로 생성했다. 허용된 새 요약1회·bank1개를 완료했다.

## 네 결과를 모두 보존한 비교

| 조건 | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP |
|---|---:|---:|---:|---:|
| A2 DP 잡음1 | 0.658184 | 0.076739 | 0.730769 | 0.080103 |
| A2 DP 잡음2 | 0.645283 | 0.074946 | 0.724648 | 0.076498 |
| DINO DP 잡음1 | 0.672935 | 0.080814 | 0.707482 | 0.085203 |
| DINO DP 잡음2 | 0.624128 | 0.067322 | 0.647386 | 0.065002 |

방법별 두 bank의 기술평균:

| 방법 | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP |
|---|---:|---:|---:|---:|
| A2 | 0.651733 | 0.075842 | 0.727709 | 0.078301 |
| DINO | 0.648531 | 0.074068 | 0.677434 | 0.075103 |

DenseNet DINO−A2 기술평균 차이는 AUROC -0.003202, AP -0.001774다. 이는 두 bank 지표의 산술 평균이며 앙상블 성능이나 잡음 모집단의 확정 기대 성능이 아니다. 두 잡음만으로 안정성·등가성·논문 우위를 증명하지 않는다.

| DenseNet 비교 | AUROC 차이 [환자95% CI] | AP 차이 [환자95% CI] |
|---|---|---|
| DP1 − DINO_DP1 | -0.014751 [-0.054676, +0.027099] | -0.004075 [-0.022046, +0.011309] |
| DP1 − DINO_DP2 | +0.034056 [-0.005796, +0.074717] | +0.009417 [-0.010036, +0.025767] |
| DP2 − DINO_DP1 | -0.027652 [-0.079732, +0.028155] | -0.005868 [-0.026980, +0.013459] |
| DP2 − DINO_DP2 | +0.021155 [-0.023752, +0.066724] | +0.007624 [-0.012075, +0.026130] |
| DINO_DP2 − DINO_DP1 | -0.048807 [-0.095828, -0.002788] | -0.013492 [-0.031818, +0.003813] |

`DP1/DP2`는 A2, `DINO_DP1/DINO_DP2`는 DINO를 뜻한다. 독립 noise 번호는 서로 결합된 짝이 아니다. 따라서 두 번호끼리만 유리하게 대응시키지 않고 네 교차 비교를 모두 보고했다. 동일2000회 환자 bootstrap은 각 고정된 bank와 실현 잡음에 조건부이며 DP 잡음/합성 seed의 전체 불확실성을 포함하지 않는다. 비교들이 bank를 공유하므로 서로 독립 검증으로 세지 않는다. CI에0이 포함된 것은 동등성 증명이 아니다.

DenseNet/V는 적응적으로 재사용한 개발 자원이다. BioViL은 A2 제작 source이지만 DINO 제작에는 사용하지 않았다. BioViL의 유지 정도와 DenseNet 전이 결과를 구분하며 어느 한 모델 결과로 전체 receiver robustness를 확정하지 않는다. 이번 반복은 기존 방법 효용의 확인이며 독립 수신자·다른 보호 예산·강한 선행 비교는 아니다.

## 구현·실행 확인

- DINO 보호 메커니즘, P-only clipping 기준 C0=2.0639112795622627/C1=1.7797955296524304, 후처리, 조건/head/projection, 학습 runtime은 기존 DP1과 동일했다. 새 target/job 연결만 확인했고 기존 수학·GPU 검사를 반복하지 않았다.
- OS 난수로 새 보호 요약1회만 생성했다. 실현 잡음/재현 가능한 DP seed를 저장하지 않았고 결과에 따른 재추출은0회다. private class 분모도 원래274좌표 query에 포함한다.
- 같은 공개 seed101 renderer/optimizer/RNG/활성 pyramid 초기값에서 새로 시작했다. 기존 bank를 이어 학습하지 않았다.
- 128장·200회·microbatch16 완료, frozen source/target 불변, 유한 loss/gradient·optimizer 갱신, 최종 PNG128장의 픽셀/label/hash 검증 통과. 학습 DINO forward 204,800장, backward 102,400장.
- 최종 PNG 고정 후 BioViL/DenseNet 각128장만 새 평가 forward. 기존 V 특징과 환자 draw 재사용. 새 Q/V pixel/model feature 추출0, Expert/Reserved 접근0.
- 독립 sklearn bootstrap 수치대조 최대오차 1.110e-16. 기존 결과의 point metric·예측·고정 source/input hash 보존.

## 실제 비용과 보호 회계

합성·저장 **35.13분**, 평가 11.17초, 준비 시작부터 비교 완료 **40.13분**이다. GPU peak allocated 2646.22MiB/reserved 2892.00MiB. bank60분·전체75분 상한 안에 완료했다. 완료시 남은 디스크 4.20GiB. 이번 bank의 중복 checkpoint만 정리했고 이전 자료는 삭제하지 않았다.

각 보호 요약은 개별(8,10⁻⁵) 설정이다. 같은 Q에 대한 기록된 요약은 이제 A2 2회+DINO 2회다. 네 요약을 함께 공개할 때 기본 합성 보수적 상한은 **(32,4×10⁻⁵)**이며, 각 방법의 두 요약만이면(16,2×10⁻⁵)이다. 더 정밀한 회계를 최적화했다고 주장하지 않는다. [Dwork–Roth, Theorem3.16](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

같은 요약에서 PNG를 만드는 후처리는 추가 private query가 아니다. 고정 공개recipe의 Q 보호 요약과 후처리에 한정한 회계이며, 과거 비DP 개발 선택·내부 감사자료·V 성능 보고서 전체를 소급해 보호하지 않는다. 외부 공개는 수행하지 않았고 공개용 PNG 묶음과 내부 감사 산출물을 구별했다. 수치 검산과 별도 유한정밀도 보안 증명도 구별한다.

## 산출물과 종료

- [사전 계약](RECEIVER_DINO_DP_NOISE_REPEAT_CONTRACT_20260924.md)
- [전체 결과](../../code_working/_reports/receiver_dino_dp8_noise_repeat_20260924_v1/campaign_result.json)
- [성능·구간](../../code_working/_reports/receiver_dino_dp8_noise_repeat_20260924_v1/evaluation/result.json)
- [누적 회계](../../code_working/_reports/receiver_dino_dp8_noise_repeat_20260924_v1/privacy_release_ledger.json)
- [보호 PNG 목록](../../code_working/_reports/receiver_dino_dp8_noise_repeat_20260924_v1/protected_png/release_manifest.json)

추가 noise/seed·A2 변경·예산 탐색·학습 연장·새 수신자·Expert/Reserved는 실행하지 않았다. 허용 작업은 완료했고 연구 단계2는 in_progress다.
