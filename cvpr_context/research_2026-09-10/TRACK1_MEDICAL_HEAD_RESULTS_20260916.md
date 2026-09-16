# 공개 의료 backbone 위의 public·pooled head: 실제 생성 비교

**최종 연구 판정: 혼합적이며, 핵심인 사적 자료의 추가 생성 효용 관문은 통과하지 못했다.** 새 특징 추출부터 비DP head 학습, 연결 검산, 고정 192장 생성, 전체 가림 판독과 의료 encoder 평가를 실제 완료했다.

공개전용 head는 기반모델보다 조건별 RAD-DINO KID 평균이 **7.98% 낮아져**, 작은 head가 실제 생성 지표를 바꿀 수 있다는 긍정적 근거가 생겼다. 그러나 사적 역할 80명을 합친 pooled는 공개전용보다 같은 KID가 **0.99% 높아졌고**, precision은 같으며 density·coverage도 낮았다. BioViL-T의 해당 prompt 유사도는 조금 올랐지만, 다른 질환 prompt와의 상대적 구별은 개선되지 않았다. **현재 결과를 사적 효용 성공으로 보고 DP solver 단계로 넘어가지 않는다.**

이 판정은 단일 split·head의 개발 비교에 한정한다. 사적 자료가 일반적으로 쓸모없다거나 환자-DP 연구 전체가 불가능하다는 결론은 아니다. 현재 큰단계 2·방향 1을 유지하며, 추가 실행은 멈춘 상태다.

## 1. 질문과 고정 비교

조건부 채택된 공개 E4 LoRA·CFG7.5를 고정하고 다음 세 방법만 비교했다.

| 방법 | 사용 자료와 연산 |
|---|---|
| Backbone | 공개640명·749장으로 학습한 고정 E4, 작은 head 없음 |
| Public head | 별도 공개32명·64장의 새 특징으로64×4 선형 head 계산 |
| Pooled head | 같은 공개32명과 사적 역할80명·320장을 동일 환자 질량으로 합쳐 같은 head 계산 |

개발 MSE는 기존40명·160장에 측정한다. 이40명은 이전 설계에서 반복 확인한 개발집합이며 새 독립 test가 아니다. public/private는 NIH 공개자료의 역할 분할이며 사적 자료만의 고유 임상정보를 입증하는 구도가 아니다. 공개환자당2장과 사적/개발환자당4장이라는 관측량 차이도 유지됐다.

E4 checkpoint, CFG, VAE, scheduler, precision, prompt 처리, head 차원, lambda, generation seed를 결과에 맞춰 바꾸지 않았다. 새 backbone 특징·prediction·residual·A/B/Q·W·witness를 계산했다. 데이터 독립 random P만 재사용했다.

## 2. 이번에 고정한 학습목표

조건부 head는 `Delta=phi W`다. guided base를 `g0=eps_u+7.5*(eps_c-eps_u)`로 두고 `r=eps_target-g0`, `X=7.5*phi`를 회귀했다. 목적함수는 환자별 평균의 spatial mean·channel sum loss에 `.001||W||²`를 더한 것이다. 실제 conditional branch에만 Delta를 더하므로 guided 출력에는 수학적으로7.5Delta가 추가된다.

학습목표가 실제 guided correction을 포함하도록 정한 것이지, MSE를 낮추면 생성 품질이 좋아진다는 가정이 아니다. 실제-noise MSE에 맞춘 보정이 유용한 guidance 효과를 약화할 가능성도 있다. 기존 CFG1 conditional-only MSE와 수치 자체를 직접 비교하지 않는다.

## 3. 보조 결과: 개발 denoising MSE

| 방법 | 개발 MSE |
|---|---:|
| Backbone | 0.4672967921 |
| Public head | 0.4479379975 |
| Pooled head | 0.4477524809 |

Public은 backbone보다4.1427% 낮다. Pooled는 public보다 상대0.04142%, 절대0.0001855165 낮고40명 중35명에서 낮았다. 환자별 pooled−public 변화 중앙값은−0.0001617400이다. **방향성이 있는 작은 재사용 개발 MSE 차이이며, 생성 효용 판단의 primary가 아니다.**

4,352개 원시 특징·분기 epsilon·target을 저장하고152명 통계를 재구성했다. 별도 축 결합·Cholesky·직접 픽셀 loss 재계산으로48,233검사PASS를 얻었다. 이 수는 통계적 표본 수나 유의성의 근거가 아니다.

## 4. 실제 생성 및 평가 결과

방법당 64장·총 192장을 한 번 생성했다. 새 16개 실제 latent를 네 prompt와 세 방법이 공유한다. 비교는 64개 cell끼리 대응되며 독립 잡음 block은 16개다. 결과를 본 뒤 checkpoint·CFG·lambda·seed·correction scale을 바꾸거나 재생성하지 않았다.

| 지표 | Backbone | Public head | Pooled head | Pooled 대 Public |
|---|---:|---:|---:|---|
| 조건별 KID 평균 ↓ | 0.442094258 | 0.406798705 | 0.410835059 | +0.004036354, 0.99% 악화 |
| Normal KID ↓ | 0.433798740 | 0.396850848 | 0.401228263 | 악화 |
| Effusion KID ↓ | 0.450389776 | 0.416746561 | 0.420441854 | 악화 |
| 혼합 KID, 기술 점수 ↓ | 0.430505826 | 0.397055009 | 0.400631919 | 악화 |
| PRDC precision ↑ | 0.6250 | 0.8125 | 0.8125 | 동일 |
| PRDC recall ↑ | 0 | 0 | 0 | 모두 낮음 |
| PRDC density ↑ | 0.38125 | 0.59375 | 0.51875 | 낮아짐 |
| PRDC coverage ↑ | 0.375 | 0.400 | 0.375 | 낮아짐 |
| 전체 64장 feature effective rank | 28.3686 | 26.4986 | 26.1855 | 조금 낮아짐 |
| 전체 64장 feature variance 합 | 57.5447 | 60.8507 | 61.2094 | 조금 높아짐 |
| Specific prompt 평균 cosine ↑ | −0.218699 | −0.122462 | −0.112794 | +0.009668 |
| Specific prompt margin ↑ | −0.000831 | −0.003019 | −0.003148 | −0.000128 |

KID·PRDC는 normal/effusion 생성 32장과 reference 40명에 대한 소표본 개발 점수다. KID 악화는 이 배치의 관측이며 모집단에서 pooled가 항상 열등하다는 확증은 아니다. Recall 0과 낮은 effective rank는 전체 영상 분포·다양성이 충분히 확보됐다는 주장을 허용하지 않는다. 두 다양성 지표도 같은 방향으로 움직이지 않으므로 단일 값으로 품질 우위를 선언하지 않는다.

Pooled−public의 BioViL 결과를 16개 shared-latent block으로 집계하면:

| 비교 | 평균 차이 | 양수 block | 기술적 bootstrap 95% 범위 |
|---|---:|---:|---|
| 해당 prompt cosine | +0.009668 | 13/16 | [+0.003777, +0.016082] |
| 해당 prompt−다른 두 specific prompt 평균 | −0.000128 | 9/16 | [−0.005656, +0.004722] |

**텍스트 유사도 상승은 실제 관측된 긍정적 신호이며 숨기지 않는다.** 다만 같은 저장 점수에서 다른 두 specific prompt의 평균 유사도도 +0.009796 상승했다. 따라서 이 상승을 지정 질환의 더 정확한 반영이라고 해석할 수 없다. 아래 조건별 margin에서도 normal은 좋아졌지만 effusion과 cardiomegaly는 낮아졌다. 이것은 임상 판독 결과가 아닌 encoder의 제한된 세 문장 간 비교다.

| Prompt | Backbone margin | Public margin | Pooled margin |
|---|---:|---:|---:|
| Normal | 0.453022 | 0.417892 | 0.428097 |
| Effusion | 0.062079 | 0.067515 | 0.059726 |
| Cardiomegaly | −0.517596 | −0.494465 | −0.497266 |

Public head도 KID·precision·matched cosine에는 긍정적이지만 평균 specific margin은 −0.002188 변했다(기술 범위 [−0.017859, +0.012327]). 따라서 **공개 head가 일부 생성 지표를 개선했다**고 말할 수 있고, 질환 정합성·임상 품질·모든 효용이 개선됐다고 말할 수는 없다.

원래 질문에 대한 현재 답은 다음과 같다. 작은 head의 생성 영향과 일부 유리한 지표는 확인됐다. 그러나 공개전용 위에 사적 자료를 추가해야 할 충분한 생성 효용은 아직 확인되지 않았다. 새 DP solver나 강한 DP 비교를 지금 시작할 근거로 쓰지 않는다.

## 5. 전체 영상

Root 한 명의 AI 판독자가 방법 이름을 가린 여덟 장의 contact sheet로 **192장 전부**를 확인하고, 방법 대응표와 정량 결과를 열기 전에 관찰 기록을 확정했다. 실험 가설을 알고 있었고 스타일로 정체를 추측할 여지는 있으므로 완전한 맹검이나 임상 판독으로 부르지 않는다.

전부 정면 흉부 형태는 식별됐지만, 해부학적 정확성이나 임상 품질 PASS라는 뜻은 아니다. 일부에는 사각형·태그 같은 하단 물체, film 경계, 과도하게 매끈하거나 거친 질감이 있다. 가림 상태에서 매 cell의 두 영상은 매우 비슷해 일관된 우열을 식별하지 못했다. 해제 후 그 두 영상이 public·pooled임을 확인했다. 두 head는 backbone의 매끈한 표현을 더 복잡한 질감으로 바꾸지만, 잡상도 추가하므로 질감 증가 자체를 품질 향상으로 세지 않는다.

아래는 원본 pixel을 수정하지 않은 **전체 192장**이다. 모든 페이지의 열은 **Backbone / Public head / Pooled head**이며 고정 seed·prompt 순서다. 첫 페이지도 결과를 보고 고른 예시가 아니다.

<details open><summary>S00–S01 · 24장</summary><img src="figures/medical_head_20260916/paired_1.png" alt="S00 S01의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S02–S03 · 24장</summary><img src="figures/medical_head_20260916/paired_2.png" alt="S02 S03의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S04–S05 · 24장</summary><img src="figures/medical_head_20260916/paired_3.png" alt="S04 S05의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S06–S07 · 24장</summary><img src="figures/medical_head_20260916/paired_4.png" alt="S06 S07의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S08–S09 · 24장</summary><img src="figures/medical_head_20260916/paired_5.png" alt="S08 S09의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S10–S11 · 24장</summary><img src="figures/medical_head_20260916/paired_6.png" alt="S10 S11의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S12–S13 · 24장</summary><img src="figures/medical_head_20260916/paired_7.png" alt="S12 S13의 세 방법 전체 대응 영상" loading="lazy"></details>
<details><summary>S14–S15 · 24장</summary><img src="figures/medical_head_20260916/paired_8.png" alt="S14 S15의 세 방법 전체 대응 영상" loading="lazy"></details>

[이름을 가린 최초 관찰 기록](../../code_working/_reports/medical_head_20260916_v1/visual_review_root.json)은 별도로 보존했다.

## 6. 구현 검산과 실제 비용

추출은4,352기록·UNet4,353calls/8,706examples(warmup1 포함),backward0이었다. 실제 추출 함수490.844초,비DP 통계·fitting19.964초,독립 저장 산술 검산16.823초다. 같은-head DP-SGD도 통계를 재사용할 수 있으므로 이 작은 fitting 시간을 backbone 학습 대비 전체 속도 향상으로 주장하지 않는다.

생성 연결은6개 새 real-noise witness의 모델 출력/feature 재실행과 offline/online 보정 비교, W=0·원경로·hook 제거 후 복원30step 검사를 먼저 수행했다. 원경로와0보정 및복원은 전체 core packet이 exact였다. 연결 단계102UNet calls/204examples·VAE3회다. 새 학습·역전파·DP는 없다.

| 실제 수행 단계 | 실행 시간 | 범위 |
|---|---:|---|
| 새 backbone 특징 추출 | 490.844초 | 4,352기록 + warmup 1 |
| 통계 재구성·비DP fitting | 19.964초 | public·pooled |
| head 산술 검산 | 16.823초 | 48,233항목 |
| 연결 검사 + 192장 생성·저장 | 635.610초 | UNet 5,862회 / 11,724examples, VAE 195회 |
| 생성 packet 독립 산술 검산 | 19.633초 | 133,976항목 |
| 의료 encoder 평가 | 23.935초 | 실제 reference 40장 + 생성 192장 |
| 점수 계산 | 0.656초 | 고정 KID·PRDC·alignment·bootstrap |
| 저장 feature 독립 metric 검산 | 1.085초 | 1,878항목 |

전체 UNet은 추출·연결·생성을 합쳐 **10,215 API calls / 20,430 examples, backward 0**이다. 추출 중 사적 역할 자료는 2,560calls / 5,120examples이며 사적 자료 접근이 없었다는 뜻이 아니다. VAE 195회에는 plain/zero/restored 연결검사의 세 decode가 포함된다.

추출 peak allocated는 3.620GiB, 생성은 3.930GiB였다. 저장된 sampling+decode 평균은 backbone 2.705초, public 2.748초, pooled 2.744초/장이다. 고정 순서로 한 번 측정한 비용 관찰이며 정밀한 속도 benchmark가 아니다. Backbone 훈련과 DP-LoRA 비교 비용은 이번에 측정하지 않았다.

검산은 새 원시 통계·수식·DDIM 전이·image hash를 검사한 것이다. UNet 전체를 독립 구현으로 다시 돌린 검증은 아니며, 새 witness 6개의 실제 모델 재실행만 포함한다. 의료 encoder는 미리 정한 실제/생성 혼합 8입력 재실행이 모두 exact였고, 저장 feature로 다른 순서의 KID·직접 거리 PRDC·SVD rank·cosine·bootstrap을 검산했다. **검사 수는 성능 표본 수가 아니다.** 이번 세 수치 검산은 최초 실행에서 통과했으며 frozen source나 tolerance를 결과에 맞춰 고치지 않았다.

코드 작성·분석·판독·문서화 시간은 위 실행 함수 시간에 포함되지 않는다. 최초 전체 예상은 45–75분이었고, 실제 작업은 2026-09-16 **22:29–23:18 KST경, 약 49분**이었다.

## 7. 해석 범위

기존 모든 evaluation·auxiliary 역할과 공개 backbone 학습 환자를 제외한 reference에서 normal/effusion20명씩 총40명을 사용한다. 과거 evaluator 개발에 사용한 환자들이며 RAD-DINO의 NIH 사전학습 이력까지 분리한 새로운 final test가 아니다.

혼합32장 KID는 기존 cubic 공식을 계산한 기술 점수다. 두 prompt가 latent를 공유하므로32개 IID 표본에 대한 불편성까지 주장하지 않는다. 조건별16장 대 real20명의 KID와 같은 비중 평균을 함께 보고한다. 보완 명세는 생성 전에 고정했으며 서로 엇갈리는 점수 중 유리한 것만 선택하지 않는다.

BioViL cosine과 specific-prompt margin은 weak text alignment이며 임상 질환 정확도가 아니다. generic은 질환 alignment 주장에 포함하지 않는다. pooled−public은16개 paired seed block으로 분석한다. bootstrap 반복을 새 이미지 수나 환자 일반화 근거로 세지 않는다. 작은 head가 이미지에서 유용한 변화를 만들었는지와 사적 역할 자료가 추가 가치를 냈는지를 구분한다.

## 8. 실행 명세와 재현 자료

- [최초 고정 명세](TRACK1_MEDICAL_HEAD_PROTOCOL_20260916.md), [생성 전 평가 해석 보완](TRACK1_MEDICAL_HEAD_EVALUATION_NOTE_20260916.md)
- [추출·학습 계약](../../code_working/_reports/medical_head_20260916_v1/contract.json), [새 특징 추출](../../code_working/_reports/medical_head_20260916_v1/extraction.json), [비DP fitting](../../code_working/_reports/medical_head_20260916_v1/fit.json), [독립 head 검산](../../code_working/_reports/medical_head_20260916_v1/head_verification.json)
- [생성 계약](../../code_working/_reports/medical_head_20260916_v1/generation_contract.json), [연결 검사](../../code_working/_reports/medical_head_20260916_v1/integration.json), [평가 계약](../../code_working/_reports/medical_head_20260916_v1/evaluation_contract.json)
- [192장 생성 완료](../../code_working/_reports/medical_head_20260916_v1/generation.json), [생성 검산](../../code_working/_reports/medical_head_20260916_v1/generation_verification.json), [평가 점수](../../code_working/_reports/medical_head_20260916_v1/evaluation.json), [평가 검산](../../code_working/_reports/medical_head_20260916_v1/evaluation_verification.json), [전체 그림 결속](../../code_working/_reports/medical_head_20260916_v1/presentation.json)

## 9. 다음 결정

**현재 구성 그대로 patient-DP 확대는 보류한다.** 공개 backbone을 다시 조정할 단계도 아니다. 이번에는 기반모델과 연결 구현을 확보한 상태에서 public/pooled의 실제 생성 차이를 직접 봤고, 그 핵심 비교가 일관된 우위를 보여주지 못했다.

다음은 20–30분의 제한된 설계 검토다. 기존 환자·조건 명부와 저장 통계에서 이 공유 선형 head가 주로 공통 도메인 보정만 배우는지, 사적 코호트가 추가해야 할 정보가 현재 비교에 실제로 있는지를 구체화한다. 현재 한 split만으로 공개자료 충분성을 증명했다고 가정하지 않는다. 공개 budget·동일 관측량 대조가 정말 필요한지, 또는 실제 자료 근거가 있는 코호트 차이와 표현 변경이 필요한지를 **새 실행 전에 하나의 질문으로 좁힌다**. 가설을 살리기 위한 임의 seed·질환·분포 변경이나 solver 목록 확대는 하지 않는다.

추가 생성 효용을 설명할 구체적 근거가 없으면 현재 작은 공유 head 후보를 중단하거나 재설계한다. 이것을 자동으로 평가 논문 성공·다른 공격 주제·CVPR 기여 확보로 바꾸지 않는다. 다음 GPU 실행 비용은 구체적 변경이 정해진 뒤 별도로 보고한다. 이 후속 검토와 GPU 실험은 이번 완료 패키지에 포함하지 않았다.

현재 큰단계2·방향1이다. 이전 공개 backbone 채택과 실패 기록, final calibration/test를 유지했다. 로컬 실행을 기록하며 원격 main 확인·push를 수행했다고 주장하지 않는다.
