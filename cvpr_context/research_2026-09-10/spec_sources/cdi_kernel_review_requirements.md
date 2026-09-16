# 2번: CDI 26개 특징 추출의 연산 일치·비용 검토 요구사항

2026-09-14. [고정 연구틀](../RESEARCH_FRAMEWORK.md)의 2번 작업이다. 이 문서는 **공개 코드의 문자 그대로인 특징 추출을 현재 backend에 연결하는 1환자 진단**의 요구사항이다. 새 GPU, 구현, feature 계산 또는 fitting은 수행하지 않았다. 기준은 X01 소스와 X01.pdf p5의 GM/NO 설명이다.

제안 범위: 기존 fit 환자14393의 E2/U2, target model_1, cached P256 posterior-mode latent를 FP32 backend로 승격, 고정 generic prompt, batch1. 실제 사용 noise_stream 수와 각 seed를 실행 전 계약에 적는다. 이는 원 논문의 VAE/model/data 실험 전체 재현이 아니다.

## 1. 정확한 26개 순서와 스칼라 정의

소스 기준 경로는 code_sources/X01이다. conf/attack/cdi.yaml:2–3와 src/attacks/features_extraction/cdi.py:23–39의 순서다.

| 0-based index | 특징 | 추출 계약 |
|---|---|---|
| 0 | denoising_loss | t100, 독립 noise 5회의 **L2 norm 평균**. MSE/RMSE가 아니다. |
| 1 | secmi_stat | t0=0,t=100,step10; 결정론적 전진 후 100→110→100 왕복의 **양의 L2 norm**. |
| 2,3 | PIA, PIAN | t200,p5; eps_zero 1회를 공유. PIAN은 N·sqrt(pi/2)·eps_zero/||eps_zero||1. |
| 4–13 | GM | t=[0,100,…,900] 순서, mask ratio .2, 아래 released-code 규칙. |
| 14–23 | multiple_loss | t=[0,100,…,900] 순서; 각 L2 norm. 한 noise를 10시점에 공유한다. |
| 24,25 | NO | final exported loss, perturbation L2. 최적화 마지막 objective와 exported loss를 구분한다. |

근거: features_extraction/denoising_loss.py:22–32; scores_computation/denoising_loss.py:15; scores_computation/secmi.py:22–24; features_extraction/pia.py:14–52; multiple_loss.py:14–28; noise_optim.py:74–87. multiple_loss.yaml의 n_repetitions=5는 현재 해당 process_batch에서 사용하지 않으므로 50회로 계산하지 않는다.

**SecMI 부호:** CDI의 원시 feature는 양의 residual norm이다. 앞선 독립 SecMI 평가의 member-positive score는 그 음수였다. 기존 음수 점수를 feature1에 그대로 넣고 원형 26개와 같다고 하지 않는다. 이 단계에서는 learned combination·scaling·label에 맞춘 부호 선택이 없다.

## 2. GM: paper 수식과 코드가 다르므로 둘을 혼합하지 않는다

- gradient는 z_t=q_t(z0,eps)에서 계산한 unsquared L2 loss를 z_t에 대해 미분한다. z0/parameter/conditioning gradient로 바꾸지 않는다. [gradient_masking.py:13–25]
- |gradient|의 상위 floor(.2·D) 좌표를 mask로 고른다. 현재 latent가 4×32×32이면 D=4096이고 k=819다. 실제 shape와 k를 기록한다. [28–45]
- **released code는 clean z0의 masked 좌표를 같은 eps로 교체**한다. 이 mixed latent를 timestep t에 직접 전달하고 추가 scheduler noising을 하지 않는다. [47–64,92–98]
- 최종 feature는 masked 좌표의 ||(eps−z0)−pred||2이며 unsquared norm이다. [68–78]
- 반면 논문 p5는 z_t의 masked 좌표 교체, eps−z_t, squared L2 표기로 설명한다. 이번 literal-code 추출에서 이를 “수정”하여 섞지 않는다. 코드 일치는 논문 수식 일치를 뜻하지 않는다.

필수 보존 증거는 timestep별 gradient/mask(또는 재검산 가능한 파일), 교체 전후 입력, masked residual과 scalar다. 독립 검산은 mask 개수·원래 값 유지/교체 값·masked norm을 확인한다. 단순히 동일 producer 함수를 한 번 더 호출한 것을 독립 검산으로 세지 않는다.

## 3. NO: 최적화 목적과 exported feature를 구분한다

- t100에서 eps를 한 번 뽑아 z_t를 만들고, unbounded delta를 0에서 시작해 SciPy L-BFGS-B maxiter=5,jac=True로 최적화한다. objective는 ||f(z_t+delta,t)−eps||2이다. [noise_optim.py:21–37,45–68]
- SciPy delta/optimizer 경로의 dtype와 model 입력 .float() cast를 기록한다. 입력 cast가 delta gradient를 끊으면 안 된다. 모델 weight를 최적화하는 실행이 아니다.
- **마지막 feature 호출에는 noise 인자가 다시 전달된다.** wrapper가 noise!=None이면 q_t를 적용하므로 실제 exported loss는 ||f(q_t(z_t+delta,eps),t)−eps||2이다. [noise_optim.py:74–82; GeneralLatentDiffusionWrapper.py:58–84]
- 논문 p5의 최종 z_t+delta 평가 및 squared norm 표기와 다르다. 이번에는 코드 동작을 보존하고, 최적화 최종 objective와 exported second-noising loss를 둘 다 기록한다. 후자가 감소하지 않았다고 optimizer가 작동하지 않았다고 판단하면 안 된다.
- maxiter5는 objective/gradient 평가5회를 뜻하지 않는다. initial/final objective, delta norm, 실제 nit/nfev/njev/status/message, objective-call loss trace, 최종 delta, 두 입력의 hash 또는 재검산 가능한 tensor를 보존한다.
- 예산상 iteration limit 종료는 자동 구현 실패가 아니다. 반대로 success 플래그만으로 gradient·feature 정합성을 통과시켜서도 안 된다. 비유한값/gradient 단절/약정과 다른 noising은 구체적 실패로 기록한다.

## 4. wrapper와 RNG에서 확인할 최소 연결

1. 실제 scheduler prediction_type=epsilon, timestep alpha, latent shape/scale, checkpoint/cache/prompt hash 및 precision을 결속한다. 원본 cached FP16 latent를 FP32로 승격하는 측정임을 명시한다.
2. noise 인자 유무를 구분한다: 없으면 이미 만들어진 latent를 직접 UNet에 넣고, 있으면 scheduler noising을 한 번 적용한다. GM mask 평가와 NO exported feature가 반대 분기를 사용함을 실제 입력으로 확인한다.
3. use_grad=True인 GM gradient/NO objective만 autograd를 켠다. 다른 경로는 no_grad이고, 바깥 inference_mode/no_grad 때문에 필요한 미분이 끊기지 않아야 한다. target parameters는 frozen, parameter .grad=None, 실행 전후 weight/version 상태를 기록한다.
4. feature/stream seed를 분리하되 family 내부 상관을 보존한다: denoising5는 새 noise5개, GM은 하나를 모든 시점·mask 단계에 공유, ML은 하나를 모든 시점에 공유, NO는 하나를 최적화와 최종 추가 noising에 공유한다. 별도 seed 체계는 원문 전체 RNG 재현이 아닌 명시적 adaptation이다.
5. 두 stream을 실행하면 stochastic families의 seed/hash를 모두 보존한다. frozen eval backend에서 SecMI/PIA/PIAN은 Gaussian draw를 사용하지 않으므로 동일 이미지의 두 stream 값은 같아야 한다. 다른 경우 원인을 기록한다.
6. raw intermediate→각 scalar→26개 이름/순서까지 별도 산술 검산으로 연결한다. L2를 MSE로 바꾸거나, prefix4의 방향을 임의 통일하거나, dtype/실행 경로 차이를 비교 없이 무시하지 않는다.
7. 미분 연결의 독립 확인이 필요하면 실제 고정 입력에서 GM/NO의 방향미분을 제한적으로 대조한다. 추가 진단 F/B는 본 추출 비용과 별도 계상하며, 큰 검증 suite나 모든 cell 반복은 요구하지 않는다. 허용오차는 dtype·reduction 검토에 근거해 사전에 기록하고 결과에 맞춰 넓히지 않는다.

## 5. analytical F/B와 실제 비용

한 영상·한 stream에서 source 기본 설정과 PIA/PIAN 공동 추출을 유지하면:

| block | forward | backward |
|---|---:|---:|
| denoising_loss | 5 | 0 |
| SecMI-stat | 12 | 0 |
| PIA+PIAN | 3 | 0 |
| GM10 | 20 | 10 |
| ML10 | 10 | 0 |
| NO | nfev+1 | nfev |
| **합계** | **51+nfev** | **10+nfev** |

NO objective가 매 호출 loss와 gradient를 함께 반환하므로 nfev가 그 경로의 F/B 수다. exported NO feature가 추가1F다. PIA와 PIAN을 따로 실행해 중복3F를 쓰면 실제 비용이 늘어나며, 원소26개라는 이유로 query26회라고 하면 안 된다.

E2/U2의 **4영상·1stream**은 204+Σnfev F,40+Σnfev B다. **4영상·2streams**는 408+Σnfev F,80+Σnfev B다. Σ는 실행된 각 NO 최적화의 실제 nfev 합이다. 함수값 탐색 횟수·로딩·CPU SciPy↔GPU 전송·저장 시간이 가변적이므로 이 수를 walltime 확정값으로 바꾸지 않는다. VAE/encoder 신규 실행, 추가 검산, checkpoint recomputation 등은 따로 적는다.

## 6. 이 진단이 증명하는 범위

통과하면 해당 1환자·checkpoint·backend에서 공개 코드의 26개 연산이 입력/미분/noising/feature-order 계약대로 실행되고, 비용·메모리와 최적화 종료 상태를 관측했다는 뜻이다. **CDI scorer 학습·집합검정·환자 확장 성능, 원문 SD 실험 재현, 정상 membership 검출, E/U 우위, 실패 원인 또는 새 기여를 증명하지 않는다.**

원코드와 paper 식 차이는 보고할 대상이며 이번 실행 중 사후 수정하지 않는다. 실행이 끝난 뒤에도 2번은 진행 중이다. 다음 fitting/환자 확장 범위는 이 연산 확인과 실제 inventory/비용을 바탕으로 별도 구체화하며 자동 확대하지 않는다.

