# 보호 단위·클리핑과 E/U 관측: 원문 신규성 경계 검토

2026-09-15. 현재 연구 2번의 한정 문헌 검토다. 새 GPU·학습·공격·통계 실행은 없으며, 아래 후보는 선택 완료된 방법이나 입증된 결과가 아니다. E는 참여 모델에서 실제 사용된 후보 영상, U는 미사용 동일 환자 영상이다. 비참여·제거 모델에서 E의 실제 image membership은 0이다.

## 1. 결론

**‘사진 단위 보호가 좋아져도 사용자 위험이 남는다’, ‘일반화·클리핑이 사용자 공통 신호를 반드시 지우지 않는다’, ‘평가가 특정 누출 원인을 놓친다’는 넓은 주장은 이미 선행이 다룬다.** 단순히 이를 의료 U로 재현하는 것만으로 새로운 기여를 주장할 수 없다.

읽은 범위에서 직접 같은 실험·정리로 확인하지 못한 좁은 질문은 다음이다. **동일한 유효 patient-DP 보장과 명시된 연산·효용 조건에서, 보호 연산의 차이가 E와 U에서 관측되는 참여 신호를 다르게 바꾸어 E 기반 보호 비교의 적용 범위를 제한하는가?** 특히 같은 환자 sampling을 두고 clip-before-average와 average-before-clip을 분리하면 한 가지 구체적인 연산 차이를 논리적으로 다룰 수 있다. 이것이 새 DP 방법이라는 뜻은 아니다. 실제 의료 모델에서 효과의 크기·방향·실용적 의미는 미확인이다.

## 2. 직접 충돌하는 원문

| 원문과 이번에 직접 읽은 범위 | 이미 다룬 내용 | 이번 후보와의 경계 |
|---|---|---|
| [Kandpal 등, User Inference, EMNLP2024](https://aclanthology.org/2024.emnlp-main.1014.pdf), §4.4/Table 2(PDF p.8), App E.5–E.7(p.25), F(pp.27–28); 위협모델 §§2–3은 앞선 검토 | 공격자가 가진 자료는 실제 학습 자료일 필요가 없다. Example-DP·중복 제거·기여량 제한·per-example/batch clipping을 평가한다. DP에서 AUROC가 낮아져도 저 FPR의 TPR가 낮아지지 않는 사례와, clipping만으로는 사용자 추론을 막지 못하는 사례가 있다. User-DP가 이 공격의 ROC를 제약한다는 이론적 관계도 설명한다. | U 감사, record-DP의 사용자 보호 한계, 지표에 따른 보호 판단 차이, clipping 실패 자체는 점유됐다. 이 원문의 DP 실험은 직접 user-DP와의 동등 보장 E/U 교차 비교는 아니다. |
| [FACE-AUDITOR, USENIX2023](https://www.usenix.org/system/files/usenixsecurity23-chen-min.pdf), App C.1–C.2/Table 5(PDF pp.17–18), 위협모델은 앞선 §4 검토 | 미사용 동일 인물 영상으로 감사하며, Fawkes 입력 변형과 DP-SGD 학습 교란 뒤에도 감사가 유지되는 결과를 제시한다. | 얼굴 인식 U에서 기존 보호가 충분하지 않을 수 있다는 실험은 이미 있다. App C.2의 Low/Middle/High 결과를 동등한 user-ε 비교 또는 새로운 E/U 순위 반전으로 읽으면 안 된다. |
| [B04, ELS/ULS](https://arxiv.org/abs/2407.07737), 로컬 v1 §2 알고리즘(pp.3–4), §3(p.5), 관련 결과·비교(pp.8–11) | 이미지/record clipping과 사용자 평균 clipping, tight user accounting, gradient diversity, 고정 연산량의 noise·효용 비교를 다룬다. | 평균 gradient·보호 단위 선택·다양성과 효용의 관계는 기여가 아니다. **§3의 Lipschitz 분석은 C를 gradient bound에 맞춰 clipping을 no-op으로 둔다.** 이를 활성 clipping이 E/U 신호를 선택적으로 억제한다는 증거로 인용할 수 없다. |
| [B05, Mind the Privacy Unit!, COLM2024](https://arxiv.org/abs/2406.14322), §§1–2(pp.2–4), §4(p.6), §6/Table 6/Fig.4(p.9) | User-wise/record-wise sampling·clipping, 자료 선택, contribution cap, 동등 user privacy에서 효용 및 clipping norm 민감도를 비교한다. 평균화 뒤 clipping이 덜 민감하다는 설명도 있다. | ‘환자 평균/선택으로 개선’ 자체는 기존 방법이다. 이 결과의 지표는 perplexity이며 E/U 공격 비교는 아니다. 원 구현의 static cap 대 dynamic sample 및 부족한 기록의 resampling도 비교 차이에 섞이므로 단순히 clipping 효과로 환원하면 안 된다. |
| [B06, Tight Group-Level DP](https://arxiv.org/abs/2401.10294), §§2.2–3.1/Thm3.1(pp.3–4) | Poisson record sampling에서 Binom(k,q) 민감도의 MoG/PLD를 이용한 group accounting을 제시한다. | 환산을 바꿔 같은 checkpoint에 더 타당한 patient 보장을 붙이는 것은 관측 점수를 바꾸는 보호 처치가 아니다. ELS를 약한 kε 환산만으로 비교해서는 안 된다. |
| [A01, Disparate privacy risks from medical AI, Nature2026](https://www.nature.com/articles/s41586-026-10688-0), 본문 pp.2,6 및 방법 pp.9–10; Fig.2 설명은 공식 원문 확인 | 환자 단위 in/out 학습 분할, 기록별 위험의 환자 max, record-DP하에서도 남는 환자 tail, 부분 자료 접근을 다룬다. | 평균 공격 약화≠모든 환자 보호, 환자 단위 평가·회계 필요는 이미 주장됐다. 환자의 실제 학습 기록을 중심으로 한 위험과 부분 기록 redaction이며, 새로운 미사용 촬영 U를 E와 교차해 보호 연산을 대조하는 실험은 이 읽은 범위에서 확인하지 못했다. |

Kandpal Table 2의 구체적 중복 경계: Enron 125M에서 non-private AUROC 88.1%, TPR@1%FPR 4.4%인데 example-DP ε=2/8/32에서는 AUROC 64.7/66.7/67.9%, TPR@1%FPR 8.8/8.8/10.3%다. 이는 해당 실험의 보고값이며 DP가 일반적으로 누출을 높인다는 정리나 E/U 순위 반전은 아니다. 이런 **지표 간 불일치의 관측 자체**를 새로 발견했다고 하면 충돌한다.

추가 영상 보호 선행 [DP-FedEmb, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_To_Generate_Image_Embeddings_With_User-Level_Differential_Privacy_CVPR_2023_paper.pdf)의 §§2.2–2.3,3.1–3.3(PDF pp.3,5–7)도 직접 확인했다. User-level clipping, private local update, public pretraining, partial aggregation과 동등 privacy의 utility 개선은 이미 있다. 이 논문의 recall@FAR는 인물 매칭 **효용**이며 MIA TPR@FPR가 아니다. 본문의 일부 강한 privacy 수치는 큰 사용자 수로 외삽한 회계와 구별해야 한다. 본문 전체 공격 재현 및 부록 전체는 검토하지 않았다.

## 3. Hartmann 선행이 막는 더 넓은 주장

[Hartmann 등, Distribution Inference Risks, SaTML2023](https://arxiv.org/pdf/2209.08541)의 §IV 도입(PDF p.5), §V-A/B(pp.7–8), App B(pp.13–14)를 직접 읽었다. 로컬 파일은 `primary_user_inference_20260915b/distribution_inference_satml2023.pdf`다.

- §V-A는 일반화 개선·추가 데이터가 유한 표본/잘못된 inductive bias에 의한 누출은 줄여도, 조건부 예측 관계 자체가 바뀌는 누출을 늘릴 수 있다고 설명한다.
- §V-B는 특정 사용자/기관의 분포 참여를 묻는 distributional membership과 IRM 방어를 다룬다. 조건부 관계의 환경 불변성 및 학습이 의도대로 된다는 가정이 필요하다. 모든 사용자 영향이 제거된다는 무조건 보장이 아니다.
- App B는 평가의 입력 변수 구성에 따라 특정 누출 경로가 존재하지 않게 되어 그 경로를 이용하는 공격의 성능을 과소평가할 수 있음을 논증한다.

따라서 **‘관측/평가 설계가 누출의 일부만 보여준다’ 또는 ‘보호가 서로 다른 신호를 다르게 바꾼다’라는 일반론 역시 새 기여가 아니다.** 반면 읽은 본문·부록에서 clipping 순서/ELS/ULS의 E/U 대조는 확인하지 못했고, 전체 PDF 텍스트 검색에도 clipping은 없다. 이 검색 결과를 전체 관련 문헌에 그 주제가 없다는 증거로 쓰지는 않는다. Hartmann의 모든 증명·실험 코드·부록을 독립 재현한 것은 아니다.

## 4. 기여로 이어질 수 있는 조건 한 가지

후보의 핵심은 **보호 단위 이름의 차이보다, 특정 보호 연산이 E에서 보이는 신호와 U에서 보이는 신호의 상대 크기를 바꾸는 조건을 설명하고, 그 조건에서 보호 비교에 무엇을 추가해야 하는지 제시하는 것**이다.

설명용 한 step에서 환자 영상 gradient를 g_r라 하면, 같은 환자를 뽑더라도 A=(1/k)Σ clip_C(g_r)와 B=clip_C((1/k)Σg_r)는 일반적으로 다르다. ELS/ULS 전체 알고리즘에는 다른 sampling·노이즈·자료선택 차이도 있으므로, 이 두 식의 대조를 원형 ELS/ULS 전체의 동일 연산 대조라고 부르지 않는다. 해당 메커니즘의 환자 민감도에 맞춘 노이즈·회계도 각각 필요하다.

다른 norm을 가진 비평행 gradient가 clipping되면 A는 영상마다 다른 가중치를 주어 방향을 바꿀 수 있지만, B는 평균 방향을 유지한 채 크기를 바꾼다. E/U query의 국소 loss 변화는 각 query gradient가 이 업데이트에 어떻게 투영되는지에 달린다. 이 때문에 E와 U의 관측 차이를 만들 **수 있는 조건**은 설명 가능하다. 그러나 clip 연산의 비가환성·gradient projection 자체는 새 발견이 아니며, 영상 특이/환자 공통 성분이 실제로 식별됐다는 뜻도 아니다. 모든 gradient가 같은 방향이거나 같은 clipping factor를 가지는 등의 경우에는 이 설명이 예측하는 방향 차이가 약해지거나 없다.

논문 수준으로 남으려면 이 구조를 통해 **실질적으로 중요한 보호 비교의 적용 범위**를 새롭게 보여야 한다. 예컨대 유효한 동일 patient 보장 아래 E 관측만으로 한 보호 선택을 U에 이전할 때 결과가 안정적으로 이전되는 조건과 그렇지 않은 조건을 구분하고, 조건별로 보정·U 자료를 포함하는 평가가 어떤 결정을 바로잡는지 입증하는 것이다. 순위 반전만을 필수 성공 조건으로 둘 필요는 없고, 기존 방법 전체가 실패해야 하는 것도 아니다. 강한 표준 patient aggregation·reference/조건별 calibration을 공정하게 허용한 상태에서 어떤 판단이 달라지는지가 핵심이다.

동일 formal ε는 실현된 모든 MIA 점수/ROC가 같다는 의미가 아니다. 반대로 낮은 U 공격 성공을 실제 전체 보호효과 또는 patient ε의 독립 인증으로 바꿀 수 없다. 이 후보는 **정의한 관측 위험에 대한 비교의 적용 범위**를 다루며, 보호효과와 감사의 관측 한계를 완전히 분리했다고 주장하지 않는다. 국소 Gaussian witness가 성립하더라도 실제 비선형 생성모델 전체 학습에서의 현상과 중요성은 별도 증거가 필요하다.

## 5. 검토의 누락과 다음 설계에 전달할 사항

- B04는 로컬 arXiv v1 원문을 읽었다. 후속 *Learning with User-Level Differential Privacy Under Fixed Compute Budgets*의 공개 제목·설명은 확인했으나 출판 최종판 전체를 재검토하지 않았다. 최종 논문 positioning 때 버전 차이를 대조해야 한다.
- User-DP/분포추론/정규화 관련 전 문헌을 조사한 것이 아니다. 사용자 정보를 단순 identity feature와 동일시하거나, Hartmann·Kandpal의 더 넓은 개념을 재명명해 기여로 삼으면 안 된다.
- 현재 프로젝트의 완료된 비DP 진단은 이 DP×E/U 상호작용을 검증한 실험이 아니다. 새 처치 실행을 이 노트가 승인·자동 예약하지 않는다.
- 우선 산출물은 같은 patient sampler의 clipping 순서 비교에서 **어떤 조건에서 어떤 관측 관계를 예상하는지**와 정당한 privacy/utility/cost 비교 조건을 적은 설계다. 이후 실제 검증 범위는 그 주장에 맞춰 정한다.
