# 2단계: 저장 학습 상태로 가능한 환자 기여 대조

2026-09-15. 정적 코드 검토와 저장 checkpoint/manifest/trace의 CPU 조회만 수행했다. 새 UNet 호출, backward, 훈련, 새 환자 평가를 실행하지 않았다. 아래 대조는 설계 제안이며 실행 결과가 아니다.

## 현재 자료로 식별되는 것과 안 되는 것

현재 model1/model2는 동일 base·LoRA 초기화, 동일 1,000개 update의 난수/시간표를 사용하지만, A/B 환자군 전체가 다르다. 따라서 두 모델의 차이는 특정 환자 한 명의 기여로 식별되지 않는다. 이미 계산한 같은 환자 E2/U2 MoFit의 방향 차이도 target 변화와 각 target에서 따로 최적화한 embedding 변화가 함께 들어간다.

새 CPU 재구성에서 확인한 사실:

- 각 manifest 912장, committed exposure 4,000회, 각 영상 4–5회 노출.
- 공통 background 256명·512장에 해당하는 2,248개 슬롯은 두 모델에서 이미지·update·batch ordinal까지 동일하다.
- 나머지 1,752개 슬롯은 각 모델의 A/B 환자 자료를 담는다. 공통 background의 입력이 같아도, 앞선 다른 update 때문에 그 입력에서의 gradient와 optimizer 상태는 이후 달라진다.
- 원래 순서 규칙과 epoch permutation을 CPU에서 재생성한 노출 원장은 두 step1000 checkpoint의 원장과 정확히 일치했다. 이것은 학습 gradient나 최종 parameter의 재현 검사가 아니다.

훈련 구현: [train_coverage.py](../../../code_working/u_patient_audit/train_coverage.py), [models.py](../../../code_working/u_patient_audit/models.py), [저장 helper](../../../code_working/u_patient_audit/train_pair.py).

## 실제 저장된 재사용 자원

각 모델에 step250, step1000이 있다. 각 checkpoint에는 adapter, optimizer, scaler, step, attempt, exposures, losses, contract가 저장된다. AdamW의 256개 LoRA parameter state에 step/exp_avg/exp_avg_sq가 있으며 LoRA 총 좌표는 1,659,904개다. 두 모델 모두 attempt=step=1,000으로 기록되어 이 실행에서 overflow 재시도는 없었다.

훈련은 FP16 base와 FP32 LoRA, FP16 autocast, batch4, AdamW lr=0.0001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, 전체 LoRA gradient norm cap=1이다. 고정된 cached posterior mode latent와 weak-label prompt를 사용하며 null-caption dropout은 없다. 난수는 update 기준 CUDA FP16 Gaussian, timestep은 update/ordinal hash이므로 CPU Gaussian으로 바꾼 재생은 원래 학습의 bitwise 재현이 아니다.

cache에는 latent 2,304장과 hidden 145종이 있다. 영상 latent는 [1,4,32,32] FP16이며 기존 평가의 FP32 연산은 이를 승격한 값이다. 원영상 재인코딩이 꼭 필요한 대조는 아니다. 저장된 cache에는 학습 parameter gradient, Hessian, step별 optimizer update vector 또는 당시 diffusion noise tensor 자체는 없다. seed로 입력을 재생성할 수 있다는 것과 저장된 gradient를 읽는 것은 다르다.

관련 입력: [cache](../../../code_working/_reports/cvpr_u_pilot_v1_001/cache/summary.json), [학습 계약](../../../code_working/_reports/cvpr_u_pilot_v1_001/training_coverage_v2/protocol.json), [M1 trace](../../../code_working/_reports/cvpr_u_pilot_v1_001/training_coverage_v2/model_1/training_trace.jsonl), [M2 trace](../../../code_working/_reports/cvpr_u_pilot_v1_001/training_coverage_v2/model_2/training_trace.jsonl).

환자14393의 실제 model1 E 노출은 다음과 같다. update는 1부터, ordinal은 0부터다.

| E 영상 | update / ordinal / timestep |
|---|---|
| E006 | 44/3/172, 380/3/108, 474/3/452, 742/1/305 |
| E002 | 83/0/281, 399/2/122, 592/2/475, 904/2/922 |

step250에는 두 E가 이미 한 번씩 들어갔다. 따라서 model1 step250에서 이후 E를 빼는 실험은 완전 nonmember를 만드는 것이 아니라 나머지 6회 기여를 제거하는 실험이다. model2에서는 이 환자의 모든 영상이 step250/1000 모두 0회다.

조건도 동일하지 않다. 이 환자의 E002는 pneumothorax, E006은 no labeled finding, U001은 emphysema, U004는 pneumothorax의 weak-label template다. E→U 전달의 차이를 환자 정체성으로만 설명하면 안 된다. label 및 DINO 유사도 통제는 필요하지만 관측된 일부 특성의 매칭이 모든 분포 차이를 제거하지는 않는다.

## 대조 1: 가장 작은 국소 parameter-gradient 전달 검사

우선순위가 필요한 경우, 새 장기 훈련보다 먼저 가능한 **내부 메커니즘 진단**이다. model2 step1000처럼 환자14393이 아직 포함되지 않은 공통 상태에서, LoRA parameter에 대한 gradient만 구하고 원래 weight를 바꾸지 않는다.

- source는 E002/E006의 실제 훈련 template loss 평균이다.
- query는 같은 환자 U2와, 이미 보유한 다른 환자 자료다. 현재 reference match는 18173과25588 두 환자를 가리킨다. 이들의 저장 두 영상씩을 쓰면 비교 query는 U2+타환자4=6장이다. reference를 훈련하거나 새 reference를 점수에 맞춰 고르지 않는다.
- 첫 진단은 사전 고정한 t140·image seed 1개·FP32 eval로 정의할 수 있다. source E는 weak-label, query는 generic으로 두고 조건 차이를 명시한다. 기존 prompt 진단과 연결되지만 동일 noise인지 실제 seed 정책을 다시 고정해야 한다.
- g_E=(g_E002+g_E006)/2이고 query x의 gradient가 g_x이면, SGD 방향의 국소 loss 변화는 -η g_xᵀg_E다. dot/norm/cosine과 같은 환자 U 평균에서 타환자 평균을 뺀 값을 전부 보고한다.
- 비용은 E2+U2+타환자4에서 **8 forward-examples, 8 backward 호출**이다. 각 gradient를 따로 저장해야 하므로 단순 batch 평균 backward 한 번과 혼동하지 않는다. 2개 checkpoint에서 보면 16F/16B다. 이는 함수 호출 수이며 원래 FP16 training step 또는 MoFit input-gradient의 실행 시간과 같지 않다.

기존 R의 gradient_dot은 16차원 conditioning coefficient에 대한 것이고, MoFit packet은 pixel/embedding에 대한 것이다. 어느 것도 이 LoRA parameter-gradient 내적을 CPU에서 대체해 주지 않는다. 새 GPU gradient가 필요하다.

이 결과가 양수여도 실제 AdamW의 유한 update 효과나 training participation 판별의 증거는 아니다. Adam momentum/preconditioning, norm clipping, weight decay와 장기 경로를 반영하지 않은 SGD 국소 설명이다. 흔한 질환/해부학적 유사성도 같은 방향을 만들 수 있다. 특히 NIH denoising loss를 사용자 분포의 정확한 log likelihood로 등치하지 않는다.

## 대조 2: 같은 시작 상태의 유한 AdamW 분기

gradient 내적의 근사 한계를 실제 변화와 대조하려면, 동일 model2 checkpoint와 **동일 optimizer state**를 복제한 임시 두 branch가 필요하다. 새 state를 기존 checkpoint에 덮어쓰면 안 된다.

예를 들어 동일 background 2장+source 2장의 batch4를 4회 사용하면 source E 두 장은 각각4회 노출된다. treatment는 환자 p의 E2, control은 사전에 정한 다른 fit 환자 q의 E2로 한다. 두 source 환자는 시작 모델에서 모두 nonmember인 group A에서 정하고, label·영상 조건 매칭 규칙을 결과 전에 고정한다. background/noise/timestep/optimizer update 수는 동일하다.

- treatment와 control 각 4 update: 총 **32 forward-examples / 8 backward 호출**. gradient checkpointing에 의한 실제 재계산 호출은 별도 계측해야 한다.
- 시작 상태와 두 endpoint에서 E2/U2/타환자4의 고정 loss를 보면 추가 **24F/0B**. 합계 **56F/8B**다.
- 같은 환자 U와 타환자 query의 변화 차이를 두 branch 사이에서 비교한다. E의 자기 loss 변화도 같이 보고한다.
- 두 source 중 하나만 새 label 정보를 쓰지 않게 하고, 실제 AdamW moments·clip·normalizer·학습 모드를 동일하게 유지한다.

이것은 p 대 q라는 **자료 대체 효과**다. q의 자료를 넣는 효과도 차이에 들어가므로 '다른 자료는 전부 같고 오직 p를 포함/제외한 효과'로 부르지 않는다. 시작 상태에 과거 cohort의 학습이 들어 있고, 4 update는 원래 1,000-update 노출 시점과도 다르다. 따라서 장기 membership 원인의 확정이 아니다.

참고로 현재 동일 환자를 대체하는 control q나 4개 update의 timestep을 아직 선택하지 않았다. 이미 본 결과를 개선하는 조합을 탐색하도록 제안하지 않는다.

## 대조 3: 원래 M1의 E 기여를 제외하는 학습 경로 대조

가장 직접적인 질문은 '원래 M1의 나머지 입력·난수·update를 유지하면서 환자14393 E가 준 학습 기여를 제외하면 E/U의 저장 baseline 점수가 어떻게 달라지는가'다.

다음 두 사양은 서로 다른 estimand이므로 먼저 선택해야 한다.

1. **Masked contribution:** 원래 912-slot 순서·batch4·분모4·noise/timestep을 유지한다. 위 8개 E 슬롯의 loss gradient만 control에서 0으로 만들고 나머지 loss는 그대로 둔다. optimizer step/momentum/weight decay도 계속 실행한다. background와 다른 환자의 노출은 동일하다. 유효 비영 영상 노출은 treatment4,000 대 control3,992다. 따라서 '총 유효 노출수까지 같다'고 쓰면 틀린다.
2. **Replacement:** 같은 8개 슬롯을 고정된 q의 두 E로 바꾼다. 두 branch 모두 유효 노출4,000이고 전체 나머지 슬롯은 같다. 하지만 p 기여와 q 기여가 함께 바뀐다. 대체 효과를 p의 순수 포함 효과라고 쓰면 틀린다.

즉 **전체 background 고정·동일 실제 비영 노출수·오직 p의 존재만 변경**을 모두 문자 그대로 만족시키는 것은 불가능하다. '동일 노출'을 동일 예정 slot/optimizer step으로 뜻하는지, 실제 loss에 기여한 record 수로 뜻하는지 구분해야 한다.

원래 최초 노출은 update44여서 완전 제외 branch의 공통 prefix는 update43까지다. 해당 checkpoint는 저장돼 있지 않다. step250/1000 weight에서 p의 gradient를 단순 빼거나 optimizer moments에서 p 성분만 제거하는 방법은 자료상 가능하지 않다. Adam 비선형 경로, clip, 이후 모든 gradient 변화 때문에 endpoint 차분이나 loss trace로 정확한 unlearning을 할 수 없다.

실행한다면:

- 동일 artifact·소스·CUDA 난수로 원형 prefix를 재생한다. 원형 step250 checkpoint의 adapter/optimizer/scaler/loss/노출과 재생 값을 비교하는 conformance gate가 필요하다. 재생 도중 update43 상태를 별도 보존한다.
- 이후 update43의 동일 상태에서 control suffix update44–1000을 실행한다. 원형 경로의 재생이 성립할 때만 이미 저장된 treatment를 재사용한다. strict full endpoint 재현까지 요구하면 treatment1,000회도 다시 실행한다.
- 250-step conformance + control957-step = **1,207 batch4 update: 4,828F/1,207B**. 원형1,000-step까지 재현 + control957-step이면 **7,828F/1,957B**다. 기존 학습 gradient checkpointing의 내부 재계산은 F에 포함하지 않은 논리 입력 수다.
- 실제 기존 학습은 model1 496.731초, model2 500.581초/1,000step였다. 같은 mixed-precision training 경로 기준 단순 환산은 각각 약10분 또는16분이고, 새 구현·검산·입출력 시간과 replay 실패 대응은 별도다. 현재 FP32 attack의 분당 비용을 훈련에 대입하면 안 된다. 새 대조의 실제 시간은 미측정이다.
- 즉시 MoFit 1,000+300을 모든 branch/query에 다시 돌리지 않는다. 먼저 사전 고정한 저비용 loss/SecMI를 same-noise로 endpoint에 적용한다. 특정 baseline만 유리한 결과를 보고 바꾸지 않는다.

원래 sampler 코드를 데이터셋 크기910으로 다시 돌리면 epoch 길이와 이후 모든 순서가 바뀌므로 위 masked-slot 대조가 아니다. 새 branch가 overflow를 만나면 retry·GradScaler 차이를 기록하고, 동일 effective update/time schedule이 지켜졌는지를 따로 검토해야 한다.

## 기존 직접 baseline과의 연결 및 제한

원래 MoFit E/U 평균·최댓값은 사진에 따라 방향이 달랐다. 같은 target 내에서 optimized embedding을 교차 적용하거나 같은 query를 여러 noise로 평가하는 저장/forward 진단은 **최적화 경로·noise 의존성**을 구분할 수 있다. 이것만으로 training cohort와 개인 기여가 분리되지는 않는다.

기여 대조는 다른 질문이다. 같은 시작 상태에서 p의 E gradient를 넣었을 때 U의 고정 baseline 신호가 다른 환자보다 더 변하는지를 조사한다. 효과가 있어도 기존 baseline이 그 차이를 안정적으로 분류하는지는 별도 문제다. 효과가 약해도 U 신호의 불가능성이나 모든 공격의 실패를 뜻하지 않는다.

위 단계에서 하나를 선택한다면, 이미 예정된 저장 반복/교차 진단이 noise·최적화 변동을 얼마나 설명하는지 먼저 확인한 뒤 **목적이 국소 기전이면 대조1, 원형 E 기여의 경로 효과이면 대조3**로 나누는 것이 낫다. 대조1의 값이 유리하다는 이유로 대조2/3을 자동 승인하거나, training exposure를 늘려서 검출 성능을 만들도록 권하지 않는다. 새 훈련은 아직 실행하지 않았고 현재 승인된 GPU 일정에 추가하지 않았다.

## 이번 CPU 재확인의 결속 정보

- M1 step1000 SHA256: ae789c32ce9ebaca3eb568fda968c28d5b9cbdf39682d7a498615b26cbb08435
- M2 step1000 SHA256: ec54c5c405de03f2ba4daa361436ab1248acb541303e99291ab32ce3a7d63962
- cache SHA256: d1aa72a3c06570b9e25bc589b778448f679ab4e9f2e67073b8413d607becf275
- cohort lock SHA256: 187fdbcc6ea8b883e6b82d97f293d7a746df5408632e6f8824a9769866e1dfc4

위 기존 입력은 앞선 frozen 검산과 연결된다. 이번에 새로 계산한 것은 manifest 정렬과 epoch permutation의 committed 슬롯/노출 대조 및 checkpoint 저장 항목 조회이며, 모델 파라미터 추론이나 새 통계 적합은 하지 않았다.
