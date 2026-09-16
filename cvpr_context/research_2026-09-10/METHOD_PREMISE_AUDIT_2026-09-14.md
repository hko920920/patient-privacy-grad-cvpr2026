# 1단계 실제 검토 — 기존 방법의 작동 원리와 현재 U 후보의 근거
2026-09-14. **원문·저자 코드·기존 의료 실험을 대조한 결과다. 새 공격 성능을 검증한 결과는 아니다.**

## 판정
**U 연구 질문은 유지한다. 현재 R을 유망한 주방법으로 인정하고 확대할 근거는 아직 확보하지 못했다.** 기존 SecMI 진단의 보류를 유지한다. 학습을 늘려야 한다거나 모든 기존 공격이 실패했다고 판단하지 않는다.

이번에 확인한 구체적인 간극은 세 가지다. (1) MoFit의 강한 결과는 현재 LoRA/U와 다른 조건이고, 논문 자체에도 약한 LoRA·의료 결과가 있다. (2) CDI에는 단순 평균보다 강한 학습형 특징 결합이 있으며, 그 환자 적용은 아직 비교하지 않았다. (3) R의 교차 사진 반응 자체는 관측됐지만, 참여 환자를 구별하는 추가 정보인지는 확인되지 않았다.

직전 [PP-Mark·ICLR 실제 사례 대조](RESEARCH_DESIGN_CASE_AUDIT_2026-09-14.md)를 적용했다. 구현 검사와 절차가 갖춰졌다는 사실을, 강한 대안보다 유용할 구체적인 연구 근거로 바꾸어 말하지 않는다.

## 1. MoFit: 무엇 때문에 좋아지고 어디서 약한가
MoFit은 먼저 실제 학습 이미지의 caption 교체 민감도가 비학습 이미지보다 크다는 관찰을 제시한다. 이어 **이미지 perturbation을 unconditional loss에 맞춰 최적화 → 그 surrogate를 설명하는 embedding 추출 → 원본 이미지에 그 embedding을 넣어 조건 불일치 반응 측정**을 수행한다. 원본에 embedding만 직접 맞추면 양쪽 모두 잘 맞아져 판별력이 약해지는 대안도 비교한다. 우리 8차원 conditioning 최적화와 같은 방법은 아니다. [원문 방법](pdfs/X12.pdf#page=5), [공식 공개본](https://arxiv.org/html/2602.22689v1).

| 조건 | MoFit AUC / TPR@1% FPR (%) | 해석에 필요한 비교 |
|---|---:|---|
| COCO SD1.4, 2,500장·150,000 steps | 94.17 / 47.00 | VLM-CLiD TPR는 50.80 |
| Pokemon LoRA | 54.35 / 0.00 | SecMI 53.50 / 1.00, PFAMI 77.50 / 1.00 |
| 의료 ROCO / Prompt2MedImage | 54.44 / 2.00 | 환자 U가 아닌 이미지 train/validation 평가 |

근거: [Table 2](pdfs/X12.pdf#page=7), [Table 5](pdfs/X12.pdf#page=10), [훈련 조건](pdfs/X12.pdf#page=16), [Table 12](pdfs/X12.pdf#page=23). PFAMI의 LoRA AUC 우위도 낮은 오탐률에서의 유용성을 입증하지 않는다.

기본 COCO 공격은 이미지 1,000회·embedding 300회 최적화다. early-stop 22.32초는 surrogate 단계만이며, 전체 공격 시간으로 쓰면 안 된다. [비용](pdfs/X12.pdf#page=21). 전체 SD 사전학습 결과에서도 확인된 memorized 이미지로 member를 교체한 조건과 원래 LAION-mi를 구별해야 한다.

코드 검토: `code_sources/X12/Eval/mia_th_COCO.py:216–288`은 같은 loaded 자료에서 scaler·가중치·threshold를 고르고 성능을 보고한다. 분리된 calibration/test가 이 경로에 보이지 않는다. 그대로 평가 절차를 복제하지 않는다. surrogate의 VAE mean과 embedding 단계의 VAE sample도 구분한다. 공식 코드 다운로드는 실행 재현이 아니다.

## 2. CDI: 우리와 다른 질문이지만 강한 비교는 남는다
CDI의 P는 사용이 의심되는 공개 이미지 집합이고, U_CDI는 같은 분포의 확실한 비공개 기준 이미지 집합이다. 우리 U는 **참여 환자의 미사용 영상**이므로 다른 의미다. CDI 원형은 P/U_CDI의 출처를 분류하는 특징 결합을 학습하고 분리된 fold의 점수로 집합 차이를 검정한다. 전체 target train manifest를 요구하지 않는다. [설정·방법](pdfs/X01.pdf#page=5), [공개본](https://arxiv.org/html/2411.12858v3).

공식 특징은 기존 MIA 4개, gradient masking 10개, 다중 timestep loss 10개, noise optimization 2개의 26차원이다. 반응·최적화·gradient·여러 특징의 학습 결합은 이미 비교 대상이다. 원형 5-fold는 환자 두 장에서 성립하지 않지만, 이것이 신규성이나 개선 근거는 아니다.

**별도 알려진 개발자료에서 CDI 특징의 per-image scorer를 학습하고, 관측 두 장을 집계한 뒤 독립 환자로 보정하는 비교**가 가능하다. 원 논문 부록도 별도 5,000+5,000 이미지에서 scorer를 학습하는 MIA를 다룬다. 같은 보조정보·학습 비용을 양쪽에 허용해야 하며, 우리 80명 fitting을 그 원래 규모의 재현으로 부르지 않는다. [부록 U](pdfs/X01_supp.pdf#page=19).

“70장”은 특정 COCO 모델 결과다. 모델별 최소 P는 70–6,000장이며 주 실험은 동수 기준 자료도 쓴다. 또한 평균 p값이나 검정력 환산 TPR를 우리 patient-FPR의 실측치로 옮길 수 없다. [규모](pdfs/X01_supp.pdf#page=15), [통계](pdfs/X01_supp.pdf#page=17).

코드 검토: `code_sources/X01/experiments/cdi_perf.py:55–90`은 scaler/logistic regression을 fold별 학습하고 확률을 출력한다. `src/evaluation/evaluate.py:89`는 양측 Welch p값과 방향을 별도 반환한다. 원문의 단측 설명과 구분한다. GM/NO의 latent·noising 차이도 이식 전에 정의해야 하며, 이 차이로 CDI 비교 자체를 제외하지 않는다.

## 3. PFAMI: 다음 비교를 고를 구체적인 이유와 구현 차이
PFAMI는 원본과 crop 영상의 상대 손실 변화를 이용한다. 절대 loss가 겹쳐도 주변 반응이 다를 수 있다는 기존 해법이다. crop은 같은 환자의 다른 실제 영상이 아니다. 원문의 SD-Pokemon 결과가 우리 의료 LoRA/U 성능의 근거가 되지는 않는다. [방법](pdfs/X11.pdf#page=6), [원문](https://arxiv.org/abs/2308.12143).

| 구현 | 실제 계산 |
|---|---|
| 원저자 X11 metric | `h=(L_crop-L_original)/L_original`; 필요시 별도 reference **모델**의 h 차감 |
| CLiD 저자의 SD 재구현 | `L_crop-L_original`; 분모와 reference-model 보정 없음 |
| 현재 R | 다른 실제 동일환자 영상의 전이에서 reference **환자 영상**의 반응 차감 |

저자 stat 설정은 10 timestep·선택 crop 하나다. 필요한 target 계산은 영상당 20 sample-equivalent UNet forward, 동일 reference 계산까지 하면 40이다. 공개 코드는 사용하지 않는 crop까지 계산하므로 그대로 실행할 때는 110/220이다. backward는 필요 없다. NN 버전에는 shadow/attack 학습 비용이 별도로 든다. [설정](pdfs/X11.pdf#page=7).

원저자 코드에서는 member=0과 `-h`를 함께 사용한다. 우리 member=1 기준 부호와 low-FPR를 명시해야 한다. 원문 식만 복사하면 부호를 잘못 해석할 수 있다.

SD 후속 코드는 512px·posterior sample·crop .825를 사용한다. 현재 256px·posterior mode 적용은 명시적인 변형이다. latent crop을 pixel crop 후 VAE encoding과 동일시하지 않는다. 원저자 Table VI 및 MoFit LoRA 표의 정확한 SD 실행 코드는 수집 저장소에서 확인하지 못했다.

근거 코드: `code_sources/X11/attack/attack_model_PFAMI.py:249–264,357–363`, `code_sources/A12_supp/CLID_MIA/mia_pfami.py:235–263,474–500`. 따라서 다음 비교를 “논문 그대로 재현”이라고 선포하지 않고, 어떤 점수식·전처리를 적용하는지 먼저 고정한다.

## 4. 현재 실험이 실제로 보여준 것
[기존 실제 재검증](U_VERIFICATION_RESULTS.md)과 [SecMI 진단](U_BASELINE_SCREEN.md)의 원시 기록을 대조했다.

- 모델당 912장·1,000 step·4,000회 영상 노출이다. 각 학습 영상은 4–5회 사용됐고 U 영상은 추가학습에서 제외됐다.
- rank-8 LoRA이며 두 모델은 같은 seed·base·공통 배경을 쓴다. 여러 환자의 포함을 동시에 바꿨으므로 독립 seed 반복이나 단일 환자 인과 실험이 아니다.
- 학습은 weak-label prompt를 항상 사용하며 null-caption dropout이 없다. 빈 prompt가 cache에 있다는 사실은 해당 prompt로 학습했다는 뜻이 아니다. MoFit의 unconditional 학습 근거를 그대로 옮길 수 없지만, 이것으로 공격 불가를 단정하지 않는다.
- 현재 실제 scheduler는 **epsilon**이다. SD2.1이라는 이름만으로 v-prediction이라고 가정하지 않는다.
- 별도 품질 영상의 loss 감소와 실제 LoRA 변화는 학습·도메인 적응의 근거다. membership 검출 양성 대조는 아니다.
- 32개 support 최적화는 모두 최종 목적값이 증가했다. 저장된 FP32 endpoint에서 동일환자 query gain은 31/32, 다른 환자 reference gain은 32/32 양수다. **전이 자체는 관측됐다.**
- 그러나 R의 U8 AUC는 .6875/.2500으로 일관된 효과가 아니다. 기존 SecMI selection40 AUC는 E .5225/.5000, U .4300/.5600이며 사전 진행 기준 미충족이다.
- 강한 특징만 넣은 분류기와 거기에 R을 추가한 동일 분류기의 비교는 아직 미실행이다. 그러므로 그 추가 이득이 없다고 입증한 것도 아니다.

E가 약한 결과는 현재 공격의 민감도·학습 흔적을 함께 점검할 이유다. **E 성공이 U 성공의 논리적 필요조건인 것은 아니며, E 실패로 U 불가능을 결론 내리지 않는다.** 기존 gate 결과를 바꾸거나 결과를 본 뒤 다른 통계로 통과시키지 않는다.

## 5. R에서 아직 입증해야 하는 부분을 수식으로 특정했다
[원 설계](EXPERIMENT_DESIGN.md)의 한 support/query 방향에서, 실제 query noise·timestep·참조 가중치를 고정한 평균 함수를 Q(a), C(a)라 하자.

```text
R = [Q(a_S)-Q(0)] - [C(a_S)-C(0)]
g_Q = grad Q(0),  g_C = grad C(0)

R = (g_Q-g_C)^T a_S + remainder
```

매끄러운 비양자화 함수에서 선분 [0,a_S]의 Hessian operator norm이 H_Q,H_C 이하이면, remainder의 절댓값은 `0.5*(H_Q+H_C)*||a_S||^2` 이하이다. 첫 normalized ascent 한 번만 쓴 경우에는 `a_S=eta*g_S_opt/(||g_S_opt||+eps)`이므로 첫 항은 **support/query gradient 내적에서 reference 내적을 뺀 것**이다. g_S_opt는 support 최적화용 noise·timestep의 미분이며 query 쪽과 구분한다. 두 방향은 각각 전개한 뒤 원래 가중치로 합친다.

이는 이번에 대조한 **조건부 수학 설명**이지, 실제 R이 선형항만으로 설명된다는 실험 결과는 아니다. 6회 반복, 명목 반경 .05, LoRA rank 8은 선형항 우세를 보장하지 않는다. 실제 FP16 양자화 오차도 따로 고려해야 한다.

기존 gradient dot/cos는 query/noise/reference 정의가 달라 이 정확한 대조를 완료한 것이 아니다. 그러므로 현재 요구되는 근거는 다음과 같이 좁혀진다.

| 필요한 주장 | 현재 증거 |
|---|---|
| 최적화가 작동하고 다른 영상에도 반응이 전달됨 | 관측됨 |
| 그 반응이 membership 때문에 생긴 환자 특이 정보임 | 미확인 |
| 단순 gradient 관계·기존 fluctuation·학습 집계가 놓치는 정보임 | 미확인 |
| 같은 정보·환자 오탐·비용에서 유용한 추가 이득이 있음 | 미확인 |

참조 영상에도 전이됐다는 이유로 참조 차감을 실패 원인으로 확정하지 않는다. 공통 성분 제거가 유용할 수도 있고 유효한 membership 성분을 없앨 수도 있으며, 현재 숫자만으로 구분되지 않는다.

## 6. 다음 작업은 무엇을 결정해야 하는가
**새 R 튜닝이나 추가 target 학습부터 하지 않는다. 다음 한 작업은 기존 체크포인트에서 PFAMI 계열의 상대 반응을 공정하게 정의해, 절대 loss가 약할 때 남는 정보가 있는지 확인하는 비교 준비다.**

선정 근거는 기존 논문에서 검증한 원본/crop 상대 반응이라는 측정 축이다. SecMI는 단순 denoising loss가 아니라 deterministic 왕복 재구성 오차이며, PFAMI는 그 고정 통계·기본 loss와 다른 반응을 측정하는 비교 후보다. 이 차이가 현재 SecMI의 약한 결과를 설명하는 원인으로 확인된 것은 아니다. 원저자 정규화식과 후속 SD의 단순 차분식이 다르므로, 먼저 그 차이를 고정해야 한다. 그 뒤 원본/crop의 같은 저장 loss에서 절대값·차분·정규화의 차이를 비교할 수 있다. 같은 raw loss에서 계산되는 지표는 별도 GPU 실행으로 중복하지 않는다.

- 기존 E/U와 같은 모델을 사용하고 U를 주질문으로 유지한다. E는 해석을 돕는 대조다.
- 전처리·crop·noise·prompt·score 부호를 결과 전에 정한다. 공개 base를 reference 모델로 쓰면 원문의 disjoint 의료 reference 모델과 다른 변형이라고 표시한다.
- 비교를 준비했다고 바로 “실험할 가치가 입증됐다”고 하지 않는다. 실제 입력 수·진행 기준·측정 가능한 효과·계산량을 정한 뒤 실행 범위를 고정한다.
- 학습을 늘려야 할 이유가 생기더라도 이미 남아 있는 250/1,000-step checkpoint 활용 가능성을 먼저 판단한다.
- PFAMI 성공도 새 방법의 기여가 아니다. 그 다음에야 **무엇을 놓치는지 확인된 강한 기존 방법**에 대하여 환자 간 전이를 설계할 근거를 평가한다. 그런 근거가 없으면 R을 억지로 살리지 않는다.

이번 단계에서 확보한 것은 기존 방법의 조건·대안 설명·현재 후보의 미확인 전제를 구체화한 판단이다. 새 환자 공격·보정/시험 평가·학습은 실행하지 않았다. 다음 실행의 시간은 이식 범위와 짧은 실측으로 다시 보고하며, 원문 query 수를 우리 장비의 확정 시간으로 바꾸지 않는다.

## 직접 확인한 출처와 실행 범위
- MoFit: X12 PDF 방법·실험·관련 부록과 COCO/Pokemon/Eval 코드. 소스 커밋 `91e4b5edc153bac84b0b4209f70d1b2b94e653b2`.
- CDI: X01 본문과 통합 supplement, feature/scoring/evaluation 코드. 소스 커밋 `dcd62258b0b3fde05d52aaecfade3b5f4c09507a`.
- PFAMI: X11 방법·실험·관련 분석, 공식 attack 코드와 CLiD의 SD 재구현. X11 소스 커밋 `f80df339e11ff49db211a455c0a455d630cf34a7`.
- 현재 연구: EXPERIMENT_DESIGN, U_EXECUTION_PLAN, train_coverage/build_cache, training_coverage_v2 기록, 기존 U 분석·FP32 endpoint·SecMI 결과.
- 원문 핵심과 최종 판단을 주담당이 직접 대조했으며 방법별 검토 및 수식·해석 검토를 병행했다. 기존 79편 전체를 이번에 새로 정독했다고 하지 않는다.
- 기존 연구 데이터·모델·점수·동결 실험 코드는 유지했다. 수정 범위는 이번 판단 문서·상태 안내·HTML 연결이다.
- 소요 약21분(20:43–21:04 KST). 처음 안내한20–30분 범위 안에서 원문·코드 대조, 독립 해석 검토, 기록·HTML 연결을 마쳤다. 새 GPU 시간은0이다.
