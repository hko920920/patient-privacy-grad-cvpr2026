# 2번 추가 원문 대조: 미사용 관측 자료로 사용자 참여를 보는 선행과 필요한 대조

2026-09-15. 연구 목표와 단계는 [고정 프레임](../RESEARCH_FRAMEWORK.md)을 따른다. 이번 추가 검색으로 원문 두 편을 확보하고 아래 선택 절을 직접 읽었다. 기존 79편 장부의 읽기 범위나 편수를 소급 변경하지 않는다.

## 실제 읽은 원문과 현재 설계에 주는 제약

**Kandpal 등, EMNLP 2024, User Inference Attacks on Large Language Models.** §3–4.1, Proposition 1, Appendix E.8의 Tables 6–7을 대조했다. 사용자의 학습 사진과 감사 자료가 다른 것에 대응하는, 사용자의 미사용 문서만으로 fine-tuning 참여를 추론하는 설정이다. 사용자별 자료의 10%를 공격 관측용으로 제외하며, pretrained reference 대비 문서 log-likelihood ratio를 평균한다. 이상적으로 분포를 정확히 학습한 모델이라는 가정 아래 사용자 기여 비중과 다른 사용자 분포와의 차이를 분석한다. Table 6에서는 기존 mean이 max/min보다 좋았다. 따라서 U 설정·여러 자료·reference 평균 자체는 새 기여가 아니다. 이 논문의 likelihood 접근을 이미지 생성 API나 단일 timestep denoising loss와 같다고 해석하지 않는다. [공식 원문](https://aclanthology.org/2024.emnlp-main.1014.pdf) · [보관 PDF](primary_user_inference_20260915b/user_inference_emnlp2024.pdf#page=3)

**Hartmann 등, SatML 2023, Distribution Inference Risks: Identifying and Mitigating Sources of Leakage.** §III-B와 Appendix B의 실험 해석을 대조했다. 한 기여자의 자료를 포함/제외한 두 학습 분포와 정확한 record 대신 그 기여자의 분포를 아는 공격자를 구분한다. 논문은 반복된 개인의 건강 측정값도 예로 든다. 동일 분포의 다른 기여자가 있는 경우와 개인 식별의 관계에도 조건을 붙인다. 관측 결과만으로 어느 누출 원인이 작동했는지 단정하지 말고, 해당 원인이 들어간/제외된 실험을 대조한다는 논리가 현재 2번에 직접 유용하다. 분류·회귀 이론의 결론을 의료 diffusion의 실측 결과로 옮기지는 않는다. [저자 공개본](https://arxiv.org/pdf/2209.08541) · [보관 PDF](primary_user_inference_20260915b/distribution_inference_satml2023.pdf#page=4)

**기존 A07, Li 등, ICLR 2022 PAIR²Struct workshop.** PDF pp.3–5를 다시 직접 읽었다. 학습한 사람의 미학습 이미지가 metric embedding에서 더 조밀한 군집을 이룬다는 관측에 centroid/pairwise 거리를 연결한다. EncoderMI를 사용자별 다수결로 바꾼 대안도 비교한다. 이 선행의 강점은 관계 특징을 쓴다는 이름 자체가 아니라, 해당 학습 목표와 실제 군집 차이에 근거가 있다는 점이다. 우리 diffusion이 같은 군집 성질을 갖는지는 별도 관측이 필요하다. Table 1/2의 accuracy를 patient-FPR 1%의 TPR로 바꾸지 않는다. [보관 원문](../pdfs/A07.pdf#page=3)

## 이번 추가 실험의 논리

다음은 위 원문의 결과를 우리에게 적용했다는 결론이 아니라, 현재 관측의 경쟁 설명을 구별하는 설계다.

1. 저장된 CDI DL의 난수 5회와 t140 신규 반복으로 **측정 변동**을 확인한다. 평균으로 개선되면 이미 있는 방법의 보완이며 새 환자 공격의 필요성은 아니다. 점수의 환자 간 총분산과 membership에 따른 작은 차이를 혼동하지 않는다.
2. MoFit의 고정된 최종 embedding을 다른 사진·recipient 모델·새 noise에 교차 적용한다. 기존 자기-embedding 대조에서 모델 변화와 최적화된 embedding 변화가 동시에 바뀐 영향을 구분한다. noise별 재최적화한 원형 공격의 성능 비교와 다르다.
3. 앞의 두 실험은 여전히 전체 A/B 학습 환자군이 다른 모델을 사용한다. 이를 직접 줄이는 대조로, 원래 M1을 재현하고 환자14393의 예정 8슬롯에서 loss 기여만 0으로 하는 control을 준비한다. 나머지 순서·난수·분모·optimizer를 고정한다. 예정 4,000슬롯은 같지만 비영 기여는 4,000 대 3,992라는 차이를 명시한다. 원형 재현 gate 실패 시 control이나 해당 인과 해석으로 넘어가지 않는다.

마지막 대조의 질문은 **그 환자 E의 학습 기여가 E/U 및 다른 환자의 고정 baseline 응답을 각각 얼마나 바꿨는가**다. 한 환자의 경로 개입 결과를 일반적인 MIA 성능, DP 보장, 새로운 알고리즘 기여로 부르지 않는다. 새 공격 설계는 그 뒤에 남는 구체적인 정보 손실을 확인해야 가능하다.

[추가 원문 취득 URL·SHA256](primary_user_inference_20260915b/acquisition.json) · [학습 경로·노출 원장의 실제 재검토](stage2_training_contrast_options_20260915b.md) · [MoFit 교차 대조 사전 해석](stage2_mechanism_candidates_20260915b.md)
