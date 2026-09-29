# 1번: 일반 공개자료 KD와 현재 평균 전달의 대응 비교

작성일: 2026-09-28  
판정: **현재 고정 조건에서는 평균 전달 방식의 추가 개발 효용을 확인했다.**  
상태: 1번 실행 완료. 2번·3번 및 추가 실험은 실행하지 않았다. 큰 단계2 유지.

## 결과부터

동일한 feature-DP1/DP2 요약·128 PNG·DINO 목표·joint 라벨 최적화에서 ResNet18 목표 생성만 일반 teacher-score KD로 바꿨다. 기존 현재 방식(MT)의 결과를 그대로 재사용했다.

| 기존 보호 요약 | 일반 KD AUROC | 현재 MT AUROC | MT−KD | 95% 대응 환자 bootstrap CI |
|---|---:|---:|---:|---|
| DP1 | 0.618654 | 0.686370 | **+0.067716** | **[+0.043047,+0.093715]** |
| DP2 | 0.627710 | 0.663404 | **+0.035694** | **[+0.006521,+0.065100]** |

두 차이의 기술 평균은 **+0.051705**, 현재 두 release에 조건부인 환자 bootstrap CI **[+0.029475,+0.074209]**다. 예측을 평균한 ensemble 성능이 아니다.

| 기존 보호 요약 | 일반 KD AP | 현재 MT AP | MT−KD | 95% 대응 환자 bootstrap CI |
|---|---:|---:|---:|---|
| DP1 | 0.070750 | 0.081264 | +0.010514 | [-0.007547,+0.022903] |
| DP2 | 0.066378 | 0.074463 | +0.008085 | [-0.001974,+0.024016] |

평균 AP 차이는 +0.009300, 조건부 CI [-0.001632,+0.018919]. **AP 점추정은 두 번 모두 개선됐으나 AP 우위는 미확정**이다.

이번 수치는 DenseNet의 영상별 AUROC/AP다. 2,000회 bootstrap은 같은 환자의 영상들을 함께 재표집하는 patient-cluster 방식이다. 환자별 평균 예측의 AUROC라고 부르지 않는다.

## 이번에 확인한 것과 남은 것

**확인한 것:** 보호 요약·이미지·공개자료·라벨 solver가 같아도, 공개 teacher 점수를 RN에 회귀시키는 정의된 KD 대조와 현재 평균 전달은 다른 목표를 만들었다. 그 차이가 현재 두 bank의 DenseNet AUROC 차이로 이어졌다. 단순한 numerical integration 통과가 아니라 downstream 개발 효용의 양성 결과다.

**아직 아닌 것:**
- 모든 KD variant, POST 전체, KIP 전체 또는 최신 방법보다 우월하다는 증거.
- source-only label solve보다의 일반적 우위. 기존 DP1 비교는 여전히 불확실하며 그 기록을 바꾸지 않는다.
- 합성영상이 공개영상보다 필요하다는 증거.
- 독립 receiver/test·새 환자 cohort·DP noise 모집단 전반의 재현.
- 평균 전달 자체의 신규 수학 원리나 CVPR 기여 확정.

이번 계약의 개발 투자 기준(두 AUROC 점차이 양수, 평균≥.01, 평균AP≥0, 평균 AUROC 조건부 CI 하한>0)을 충족했다. 이는 정해진 내부 대조에서 다음 검토를 진행할 근거이며 paper-level 완료 기준이 아니다.

## 무엇을 같게 두었나

실행 승인: 사용자가 1번 단독 예상30~60분을 확인한 뒤 “진행”이라고 지시했다.

- 같은 기존 DINO feature-DP1/DP2, 각 ε8/δ1e−5.
- 각 release에 대응하는 기존 128장 PNG. 새 이미지/재합성 없음.
- 같은 DINO K4 source, P projection, ResNet18 공개 특징과 전처리.
- 공개 P의 class별 환자 균등 무게: 각 class 총0.5, 환자 안 방문 균등.
- 같은 DINO 목표 classifier.
- 같은 DINO/RN 두 모델 joint 라벨 목적, 모델별 공개 목표 RMS 정규화, 각0.5 비중.
- 같은 ridge0.1, label bounds[-1,1], solver tol1e−12/max_iter500.
- 같은 기존 64개 -1/64개 +1 hard-label anchor 및 eta=.001||A||²/128.
- 최종 학습도 균등1/128·frozen-feature ridge0.1·intercept 없음.
- 두 KD 라벨을 함께 seal한 뒤 V 평가. DenseNet은 construction loss에 포함하지 않음.

**전체 1·2·3 계획의 zero-centered regularizer는 사용하지 않았다.** 그것은 공개 carrier 비교에 필요한 통제였으며, 1번 단독에서 원래 계산을 바꾸지 않도록 제외했다.

RN 목표가 다르면 해당 RMS와 공식으로 산출한 eta도 달라진다. 동일 정규화·eta 규칙을 유지했으며 eta의 실제 숫자까지 같다고 주장하지 않는다. 이 비교는 두 전체 목표 생성 구성의 차이다.

## 정확한 KD 대조와 수식상 동치의 범위

P의 영상 특징 X=DINO, Y=RN, 공개 가중행렬 W라 하면:
\[
w_D=(X^\top WX+0.1I)^{-1}(\tilde\mu_1-\tilde\mu_0)/2,
\]
\[
q=Xw_D,\quad
w_R^{KD}=(Y^\top WY+0.1I)^{-1}Y^\top Wq.
\]

q에 sigmoid, clipping, temperature를 적용하지 않았다. KD에도 공개 class labels를 가중치에 사용할 수 있게 했다. 현재 MT는 기존의 class-clipped 공개 affine maps로 평균을 전달한 후 같은 공개 second moment와 ridge0.1로 RN 목표를 만든다.

이 두 목표는 실제 공개 P 예측에서 같지 않았다. RN 목표 예측의 상대 L2 차이는 MT를 분모로 DP1 0.714683, DP2 0.820985였다. 크기 차이를 포함한 기술통계이며 원인 증명이나 어떤 목표가 더 옳다는 증거가 아니다.

한편 **현재 MT는 특정 signed public labels를 붙인 ridge와 정확히 같다.** 그 동치는 이번에도 계수 최대 오차9.77e−15로 확인했다. 따라서 그 signed-label 방식은 MT의 별도 경쟁 방법이 아니며, 그 연산을 새 원리로 세지 않는다.

이번 결과가 보여주는 것은 “모든 공개자료 KD와 다르다”가 아니라 **정의한 통상적 teacher-score KD보다 현재 target recipe가 이 개발 설정에서 더 유용했다**는 것이다. 공개 회귀 regularization·class별 clipping·centering 등의 어느 요소가 차이를 만든 것인지는 이번 비교 하나로 분리하지 않았다.

## Construction 모델의 참고 결과

| 기존 요약 | 일반 KD ResNet18 AUROC/AP | 현재 MT AUROC/AP | ΔAUROC [95% CI] |
|---|---:|---:|---|
| DP1 | 0.663166 / 0.075601 | 0.698797 / 0.083631 | +0.035631 [+0.015178,+0.057478] |
| DP2 | 0.670401 / 0.075730 | 0.707042 / 0.084556 | +0.036641 [+0.006891,+0.065889] |

ResNet18은 목표 생성과 label solve에 참여했다. 이를 독립적인 전이 성공으로 추가 계산하지 않는다. AP는 DP1만 양의 조건부 차이 구간이며 DP2는0을 포함한다.

KD 자체의 bounded label optimization은 두 경우 정상 종료했다.
- DP1 full objective: 1.290161 → 0.059667.
- DP2 full objective: 1.127619 → 0.059882.
- KD 공개 RN relative RMSE: 0.225447 / 0.219354.
- 비교 목표가 서로 다르므로 MT와 KD의 loss 숫자를 직접 효용 순위처럼 비교하지 않는다.

## 검산·보존·읽기 범위

새 비교에 필요한 경로만 검산했고 이전 전체 준비 검사는 반복하지 않았다.

| 검사 | 최대 오차 / 결과 |
|---|---:|
| KD ridge vs 독립 augmented least squares | 2.78e−16 |
| KD normal equation residual | 7.81e−18 |
| MT vs signed-public-label ridge | 9.77e−15 |
| 기존 MT 라벨 재현 | 1.99e−12 |
| 기존 MT target coefficient 재현 | 2.78e−17 |
| 기존 MT V prediction 재현 | 1.39e−16 |
| KD primal/dual ridge V prediction | 2.02e−16 |
| bootstrap0/17/1999의 sklearn 독립 검산 | 1.11e−16 |
| 기존 PNG256개 SHA / 라벨 CSV float round-trip | 통과 |
| construction의 V 접근 없음 / 모든 평가가 seal 뒤 | 접근 로그 확인 |

기존 MT 결과·라벨·PNG·코드·계약은 변경하지 않았다. 새 label packet은 KD 두 개뿐이며, MT 두 번 재계산은 동일성 검사로만 사용했다. 새 source-only 대조·public128·geometry/nullspace 탐색은 실행하지 않았다.

## 실제 비용과 환경 오류

- 라벨 구성·수치 검산·CSV seal: **1.782초**.
- cached V 평가·2,000회 bootstrap·독립 검산: **9.907초**.
- 수치 worker 합계 약11.69초. 코드 작성·입력 확인·문서화는 별도다.
- 계약 동결 15:38:50 KST, 평가 완료15:40:11 KST. 이81초에는 환경 오류 수정과 도구 연결 시간이 포함된다.
- 전체 작업 시간은 위 worker 시간으로 표시하지 않는다. 초기 코드/입력 확인은 계약 동결 전이다. 기록 가능한 첫 명시적 시각은15:32:27 KST이며 최종 문서화 시각과 함께 runtime_summary.json에 기록한다.
- scope 검증 시 추가 폴더 약1.95MiB. 모델 다운로드·모델 forward·픽셀 학습은0.
- 첫 시도는 threadpoolctl의 Windows 라이브러리 조회 오류로 **데이터 읽기/라벨 생성 전에** 멈췄다. 원 코드·계약·오류를 attempts/00_threadpoolctl_startup_failure에 보존했다.
- 런타임 threadpool 조회 대신 BLAS 환경변수로4스레드를 지정했다. objective·tolerance·target·data·계수는 바꾸지 않았다. 변경 해시는 계약에 기록했다.

## 보호와 평가 경계

기존 두 보호 요약과 P만으로 새 라벨을 만든 후처리다. raw Q 접근·새 DP 요약/noise는0이고, 여섯 기존 release 공동 공개의 기본 상한48/6e−5는 그대로다. 새 라벨 산출물은 각각의 원래 feature release에 의존한다.

과거 adaptive development 전체나 V를 이 DP 보장으로 보호했다고 주장하지 않는다. 이번 예측과 환자 식별자는 로컬 비공개 평가 산출물이며 외부 업로드하지 않았다. Expert/Reserved는 열지 않았다.

## 산출물과 다음 상태

- [실행 계약](contribution_kd_control_20260928_v1/EXECUTION_CONTRACT.md)
- [전체 결과 JSON](contribution_kd_control_20260928_v1/results.json)
- [구성·수치 검산](contribution_kd_control_20260928_v1/construction_results.json)
- [두 라벨 동결 기록](contribution_kd_control_20260928_v1/label_seal.json)
- [DP1 KD 라벨](contribution_kd_control_20260928_v1/label_packets/DP1_KD_soft_labels.csv)
- [DP2 KD 라벨](contribution_kd_control_20260928_v1/label_packets/DP2_KD_soft_labels.csv)
- [construction 코드](contribution_kd_control_20260928_v1/prepare_kd_control.py)
- [평가 코드](contribution_kd_control_20260928_v1/evaluate_kd_control.py)

**1번은 긍정적인 개발 결과로 완료했다.** 다음 판단 대상인 다른 모델 목표의 필요성(2번), 공개영상 carrier 대조(3번)는 별도 범위로 남겨둔다. 이번 양성 결과를 이유로 새 반복·새 요약·최종 평가를 자동 실행하지 않았다.

