# 방향2 재설계: 보호판단을 바꾸는 평가 연산 후보

작성일: 2026-09-16. 상태: 원문 대조를 거친 **조건부 연구 제안**. 신규 모델 실행·통계 튜닝 없음. 이전 순차 allocator v0의 실패는 이 분야의 공백을 증명하지 않는다. 기존 E/U·clipping 실험의 결과를 새 제안의 증거로 전용하지 않는다.

## 판단

현재 가장 구체적인 운영 후보는 **교차 증류 방어에서 어느 데이터 경로를 차단해야 실제 보호가 좋아지는지 판별하는 평가**다. 그러나 ‘다른 사진을 통해 teacher 정보가 전달된다’, ‘private student 입력도 샌다’, ‘student/teacher를 나누어 평가한다’는 모두 직접 선행이 있다. 이 제안을 지금 CVPR 기여가 확보된 방향으로 부르면 안 된다.

남길 수 있는 주장은 더 좁다. 기존 보호 연산의 전제를 확인한 뒤, 그 전제가 성립해도 남는 누출 경로를 분리하고, **환자 단위 teacher 분할만 고칠지, student 입력 처리까지 바꿀지**라는 실제 수리 선택을 예측·검증하는 것이다. 단순 patient split 비교나 알려진 증류 취약성의 의료 재현만 나오면 독립 논문 후보로는 탈락한다.

## 1. 먼저 제거해야 하는 이미 알려진 주장

| 생각하기 쉬운 주장 | 직접 선행과 확인 위치 | 이번 설계의 처리 |
|---|---|---|
| 방어에 맞춰 공격을 다시 적응시키면 평가가 달라진다 | [Aerni et al., 2024](https://arxiv.org/html/2404.17399v1), 본문 방어별 평가 및 App. B.3 | 기본 비교 조건이다. 새 기여가 아니다. |
| member/nonmember 분포 차이와 같은 모델에 따른 간섭을 분리하자 | [Membership Inference Attacks from Causal Principles, 2026](https://arxiv.org/html/2602.02819v1), §§3, 5–6, App. A.2 | causal AUC·TPR 및 간섭을 이미 정의했다. 원인이라는 이름만 바꾸지 않는다. |
| 기존 모델만으로 분포 편향을 보정해 DP를 감사하자 | [Privacy Auditing with Zero (0) Training Run, 2026](https://arxiv.org/html/2605.14591v1), §§4–6, 9 | propensity 보정과 그 추정 한계까지 직접 다룬다. 단순 zero-run 보정은 탈락한다. |
| 필터가 희귀 사례를 지워 MIA가 약해지면 보호인지 데이터 손실인지 보자 | [Revisiting the Provable–Auditable Privacy Gap of DP-SGD, 2026](https://arxiv.org/html/2608.28934v1), App. B, C, F–G | App. B는 소수집단 삭제율·집단 효용·canary 위험을 이미 함께 검사한다. 이를 새 평가 원리로 주장하지 않는다. |
| 최종 train 밖으로 탈락한 자료의 사용도 확인하자 | [CoLA, ACL 2026](https://aclanthology.org/2026.acl-long.733.pdf), 초록·서론의 training membership/selection participation 구분 | 선택 참여와 학습 참여를 나누는 것 자체가 이미 선행이다. |

위 자료는 각각 다른 위협모형과 보장 범위를 갖는다. ‘이미 논문이 있다’가 현 의료 방어의 실제 행동을 알려주는 것은 아니지만, 일반적인 문제 제기만으로 신규성을 주장할 수 없게 한다.

## 2. 구체 후보: 교차 증류 방어의 보호 경로를 분리해 수리 대상을 선택

### 실제 보호판단

데이터 제공자는 여러 사진을 가진 사람의 자료로 diffusion teacher/student를 만들려 한다. 교차 증류는 특정 사진을 보지 않은 teacher의 반응으로 그 사진의 student target을 만든다. 여기서 결정해야 하는 것은 ‘증류를 쓰자’가 아니라 다음이다.

1. 사진별 분할을 사람별 분할로 바꾸면 충분한가?
2. teacher가 그 사람 전체를 보지 않아도, 그 사람의 원본 사진을 student 입력으로 반복 사용하면 추가 보호가 필요한가?
3. 공개·독립 reference로 바꾸거나 student 단계에 보호를 주는 비용을 지불할 근거가 있는가?

질문은 E/U에 한정되지 않는다. 소유자가 가진 자료와 정확히 같은 사진, 같은 사람의 다른 사진, teacher/student 학습 역할은 각각 분리 기록한다. 이미 알려진 patient-level 공격을 새 공격으로 포장하지 않는다.

### 원래 연산과 가장 가까운 반론

- [DualMD/DistillMD](https://arxiv.org/html/2410.16657v1), §3.2 Eq. (6), §3.3 Eq. (7), Algorithm 2: 서로 다른 shard를 teacher의 test 자료처럼 취급하고, private 원본에서 만든 noisy input을 반대 teacher로 평가해 student를 훈련한다. Eq. (6)은 가정이며 임의의 관련 사진에 대한 독립성 정리가 아니다.
- [SELENA](https://www.usenix.org/system/files/sec22-tang.pdf), 인쇄 pp.1434–1435, §§4–5: sample을 보지 않은 submodel만 사용하고 원래 train 자료로 self-distillation한다. 논문은 제한된 공격군의 결과와 보편적 DP 보장을 명시적으로 구별한다.
- [Knowledge Cross-Distillation](https://arxiv.org/html/2111.01363v3), §§2.3, 3.1, Algorithm 2: private reference를 재사용하면서 반대 shard teacher를 쓰는 선행이다. 단순 cross-fitting 수리는 이미 알려졌다.
- **Aerni App. B.3**는 near-duplicate를 다른 submodel이 본 효과가 전체 train을 query하는 self-distillation을 통해 학생에 남는 것을 직접 조사한다. 따라서 ‘같은 환자의 다른 사진이 teacher에 있다’는 하나의 관측만으로 차별화되지 않는다.
- **[Students Parrot Their Teachers, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/8b07d224a643b02e7571e083578a86d2-Paper-Conference.pdf)** §3.1, App. D.1–D.2는 private teacher / private student / self-distillation을 분리한다. 비민감 teacher와 민감 student 자료, teacher 접근 지식별 공격을 이미 평가한다. 학생 입력의 위험 존재 자체도 신규성이 아니다.

### 두 diffusion 방어를 같은 ‘보호’로 뭉뚱그리면 안 되는 이유

| 대상 원문 | 실제 보호 연산 | 원문이 다루는 보장·측정 대상 | 이번 평가가 추가로 물어야 할 것 |
|---|---|---|---|
| [Diffusion Soup, ECCV 2024](https://arxiv.org/html/2406.08431v1), §§4–5.1, Eq. (4–5), Prop. 3, §6.2 Anti-Memorization/Fig. 6 | 공통 초기값에서 shard별 fine-tuning한 **가중치를 평균**한다. Student private-input 재훈련 단계가 없다. | 지역 Taylor 근사·score/sampling 조건하의 geometric-mean 설명과 sample의 safe 모델에 대한 NAF; 실험은 Pokemon 원본–생성물의 CLIP 거리 및 이미지 재현이다. 이는 환자 membership ROC 또는 보편적 patient DP와 동일하지 않다. | 보호단위 전체를 보지 않은 구성요소가 실제 존재하는가? 생성물 copy 감소와 같은 접근권의 membership 위험 감소가 함께 움직이는가? 두 지표가 다르다는 일반 사실만으로는 신규성이 아니다. |
| [DualMD/DistillMD](https://arxiv.org/html/2410.16657v1), §§3.2–3.5, 4.1–4.3 | DualMD는 두 모델의 denoising 단계를 교대로 사용한다. DistillMD는 반대 teacher의 target으로 private noisy input의 student를 학습한다. | SecMI/PIA white-box, text-guided black-box의 AUC·TPR@1%FPR 및 SSCD/CLIP. 단일 포괄적 DP 정리가 아니다. 조건부 생성에서는 원문도 prompt 다양화 필요성을 확인한다. | teacher 경로와 student 입력 경로를 구분해야 한다. Soup에 존재하지 않는 student 경로를 Soup 실패 원인으로 설명해서는 안 된다. |

이 표가 제안하는 추가 연구는 NAF를 DP로 바꾸어 부르거나 두 논문의 주장을 억지로 동일하게 만드는 것이 아니다. 원문의 **실제 공개 객체와 명시한 보장**을 각각 존중한 뒤, 데이터 제공자의 더 강한 사람 단위 보호 요구에 어떤 조건이 추가되어야 하는지 따진다. 원문이 처음부터 주장하지 않은 환자 보장을 충족하지 못했다고 원문 정리를 반박한 것처럼 쓰지 않는다.

추가 직접 차단선: SELENA App. A.3, Remark 7(인쇄 p.1450)은 correlated points에서 일반적인 multi-query 보장을 못 한다고 명시하고, 가까운 nonmember query 실험도 한다. 따라서 ‘의료영상은 상관되어 있다’는 이유만으로 새로운 실패 원인을 발견했다고 주장할 수 없다.

### 더해야 하는 정보 — 실제 입력 명세

최종 MIA 점수만으로는 아래 경로의 차이를 알 수 없다. 감사용으로만 다음 정보를 추가한다.

- 보호단위의 teacher 학습 원장과 student 입력 원장. 실제 신원이 확실한 기존 patient ID를 우선 사용하고, 불확실한 ID 문제를 임의로 만들지 않는다.
- 각 student 입력에 사용한 teacher ID·checkpoint, teacher가 그 사람의 다른 자료를 보았는지, noisy input과 target 생성 규칙.
- 공개 teacher/reference가 후보 자료에 독립적이라는 출처. ‘공개 다운로드 가능’만으로 pretraining membership을 모른 채 독립이라고 선언하지 않는다.
- 같은 배경 자료를 둔 감사용 teacher 참여 개입과 student 입력 참여 개입. 이 정보는 내부 평가의 특권적 참값이며, 실제 외부 공격의 입력으로 주지 않는다.

### 평가 연산

후보 보호단위 p에 대해, 배경과 학습 정책을 고정하고 다음 두 개입을 교차한다.

\[
T_t=\operatorname{TrainTeacher}(D_{-p}\cup tD_p),\qquad
S_{ts}=\operatorname{Distill}(T_t,Q_{-p}\cup sQ_p),\quad t,s\in\{0,1\}.
\]

교차 teacher 선택 규칙도 명시적 정책의 일부다. 어떤 셀을 만들면서 우연히 다른 환자 teacher 접근까지 바꾸었다면 해당 비교를 단독 경로 효과로 해석하지 않는다. 정확한 원형 방어를 먼저 재현하고, 이 4셀은 원형과 분리된 감사 개입으로 둔다.

각 셀에서 기존 강한 공격을 **그 셀의 방어에 맞춰** 학습·보정한다. 최종 score 차이 하나를 privacy loss로 부르지 않는다. 동일 보호단위·접근권·오탐률에서의 공격 분포와 utility를 비교하고, 별도로 고정된 공격을 통한 paired response는 설명용으로만 쓴다.

비교하는 수리는 세 가지다: 원형 사진 분할, 사람 단위 teacher 분할, 사람 단위 분할에 student 입력 보호를 추가한 방식. 마지막은 공개·독립 reference와 student 단계 DP 등 기존 적절한 수리를 먼저 둔다. 비용은 teacher 수·재훈련·student training·reference 확보를 포함한다. 어떤 방식이 정답인지는 사전에 단정하지 않는다.

핵심 산출물은 ‘teacher-out이니 안전하다’ 같은 하나의 설명을 **어떤 경로가 닫혔고 무엇은 평가되지 않았는지**로 바꾸고, 관측된 잔여 경로에 맞는 수리가 실제 공격 위험과 효용을 개선하는지 확인하는 것이다. 그래프를 그리거나 경로 이름을 붙이는 것만으로는 기여가 되지 않는다.

### 필요한 관측과 반증

| 관측 | 허용되는 결론 | 기여 판정 |
|---|---|---|
| 사람 단위 teacher 분할만으로 기존 공격 위험이 충분히 감소하고 student 경로 추가 수리의 이득이 없다 | 기존의 적절한 분할이 이 설정을 해결한다 | 새 보호·평가 원리 주장은 약해진다. 알려진 수리의 적용 결과로 남긴다. |
| 완전한 teacher-out에서도 student 입력 참여에 따른 위험이 남고, 추가 수리가 같은 효용에서 이를 낮춘다 | 이 특정 방어에서는 teacher 제외만 확인한 평가가 수리 결정을 잘못할 수 있다 | Parrot의 일반 현상보다 더 나아간 diffusion의 정량 조건·수리 선택 예측이 있어야 후보를 유지한다. |
| teacher/student 두 경로가 각각 약해 보여도 결합 조건에서 강한 위험이 나온다 | 분리된 단독 위험만으로 원형 전체의 위험을 예측할 수 없다 | 상호작용의 반복성·기제와 이를 바로잡는 수리 기준까지 필요하다. 단순 4셀 차이는 인과 일반법칙이 아니다. |
| 효과가 calibration·공격 재학습·near-duplicate 통제로 사라진다 | 강한 기존 평가가 충분하다 | 새 평가법의 이득으로 주장하지 않는다. |

### 최소 판별 검증 — 실행하지 않음

이미 있는 원형/학생 공개 artifact가 확보되는지부터 확인하고, 없으면 이 후보의 검증에는 새 teacher/student 학습이 필요하다고 명시한다. 현재 M1/M2·mask 모델은 DistillMD가 아니므로 바로 전용해 방어의 실패를 주장할 수 없다.

첫 판별은 수십 공격×모든 조건 전개가 아니라, 원형의 positive control을 확보한 뒤 **사람 전체를 제외한 teacher와 동일 private student 입력**이라는 셀을 확인하는 것이다. 여기서 기존 강한 공격이 무엇을 잡는지 먼저 본다. 그러나 한 셀에서 leakage가 검출되는 것 자체는 이미 선행이 예측하므로 논문 성공이 아니다. 최종 유지 여부는 미리 정한 수리 선택이 원형 및 표준 사람 분할보다 실제 비용–효용–위험 판단을 개선하는지로 닫는다.

## 3. 공격·DP 감사 도구의 역할

MoFit·SecMI/PFAMI·Quantile은 이미지별 기존 신호, CDI와 강한 단순 집계는 다수 관측을 결합하는 대안, FSCA와 UserInference/FACE-AUDITOR는 사람 단위 판단의 직접 선행이다. 이들이 teacher/student 경로를 자동으로 식별해 주지는 않지만, 그렇다고 기존 공격이 실패한다고 가정해서도 안 된다. 새로운 입력 또는 접근권을 우리 방법만 쓰지 않도록 한다.

one-run/zero-run DP 감사는 정해진 알고리즘과 adjacency에 대한 경험적 하한을 다룬다. teacher 보장을 student가 단순 후처리로 물려받는지는 private student 입력 재접근 여부에 달려 있다. 이는 표준 DP의 후처리 조건이며 새로운 정리가 아니다. DP 적용 수리는 teacher·student·reference 생성 전체에서 환자 단위를 일관되게 회계해야 한다. 관측 공격의 약화로 formal ε를 줄였다고 주장하지 않는다.

## 4. 이번 검토에서 권하지 않는 별도 후보

불확실한 여러 기관 ID를 가능한 환자 partition으로 다루는 linkage-robust audit도 구체화할 수 있으나 지금 최우선으로 권하지 않는다. [ULDP-FL](https://arxiv.org/html/2308.12210v3)은 여러 silo에 걸친 사용자 DP를 이미 직접 다루며, 현재 데이터의 patient ID 오류가 확인된 것도 아니다. 단순 group-DP 상한과 불확실성 구간을 붙이기 위해 새 환경을 만드는 위험이 크다.

zero-run 보정, 필터링 소수집단 위험, 별도 student 입력 위험 등 이미 선행이 충분한 주제를 추가 후보 수를 채우려고 독립 기여처럼 나열하지 않는다.

## 최종 상태

구체적인 평가 대상·추가 정보·연산·수리 선택·반증 조건은 정리했다. **새 평가기여의 필요성은 아직 실증되지 않았고 가장 가까운 선행의 충돌도 강하다.** 따라서 이번 제안은 현재 실행가능한 연구 질문 후보이며, 이미 의미 있는 CVPR 기여를 찾았다는 결론은 아니다. 남은 논리적 핵심은 ‘이 정보를 추가해서 기존의 표준 수리로는 못 내리던 더 나은 보호 결정을 실제로 내릴 수 있는가’다.
