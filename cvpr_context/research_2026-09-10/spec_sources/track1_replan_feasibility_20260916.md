# 방향 1 재계획: 사적 자료의 추가 효용을 확인하는 최소 패키지
2026-09-16. 최신 저장 결과·코드·manifest/schema를 읽은 실행가능성 메모이다. 이번에는 새 모델을 계산하거나 GPU·학습·튜닝을 수행하지 않았다. 기존 계약·결과를 수정하지 않는다.

## 1. 먼저 바꿔야 하는 질문
다음 질문은 “보호 모델이 처음의 일반 base를 이기는가”보다 **“이미 사용할 수 있는 공개 자료32로 적응한 모델에 사적80을 추가하는 것이 무엇을 더 주는가”**이다.

현재 development40의 full64 MSE는 base .1796449533, public-only32 .1740411864, private-only80 비보호 .1740069294, private-only DP-SGD .1743888460, SSP .1788939244다. Static16도 같은 정성적 순서이며, 네 DP 조건 모두 공개 baseline을 이긴 잡음 반복은 0/16이었다. 이는 저장된 현재 비교 결과이다. 강한 선행 전체의 실패·모든 사적 추가 효용의 부재를 의미하지 않는다.

**Public-only32와 private-only80은 공개+사적112를 함께 학습한 비교가 아니다.** 후자의 결합은 추정 분산과 목적의 분포를 바꿀 수 있다. Private-only 해를 결합 모델의 held-out 성능 상한으로 부르는 것은 틀리다. 또한 비보호 ridge는 해당 finite training objective의 최적해이지 held-out MSE 또는 생성 품질의 수학적 상한은 아니다.

## 2. 확인한 재사용 자산
- `code_working/_reports/frozen_residual_capacity_20260916_v1/patient_stats.npz`: patient_ids/splits, train80/eval40. A_full[120,64,64], B_full[120,64,4], Q_full[120], static16 대응 통계, time4 통계까지 FP64로 저장되어 있다.
- `code_working/_reports/frozen_residual_public32_20260916_v1/patient_stats.npz`: 독립 공개32, split=public, 같은 projection·basis·정답의 full/static A/B/Q. 환자당2장×8회이며 private/eval은 환자당4장×8회다.
- 두 파일은 환자 단위 통계이므로 공개·사적 결합은 **환자 가중치**로 한다. 사진 수2:4를 이유로 공개 환자를 절반 가중하지 않는다. 두 영역의 사진 수 차이는 통계 추정 분산의 한계로 따로 남긴다.
- `public32/dp_calibration_v1/public_only_models.npz`: 공개 ridge W0.
- `frozen_residual_patient_dp_20260916_v1/controls.npz`, `models_manifest.json`, `evaluation_mse.npz`: 기존20개 control, 64개 DP 모델, 환자별 MSE.
- `frozen_residual_head/dp_mechanisms.py:164`의 `poisson_sgd(...,initial_weights=...)`는 공개 W0를 이미 지원한다. 기존 함수에 공개 평균 gradient를 함께 더하는 옵션은 없다.
- `run_capacity.py:199` 이후의 tap은 conv_out 직전 activation이다. 추출 코드는 있으나 frozen residual head를 실제 sampling trajectory에 적용하는 generation wrapper는 아직 없다.
- 기존 `u_patient_audit/generate_quality.py`에는 fixed prompt/seed·DDIM30·CFG7.5·256px 로딩과 grid 저장 코드가 있으나, LoRA 품질 진단용이다. 그대로 호출하면 새 head를 검증하는 것이 아니다.

결합 head와 head 수준 DP 대안의 학습·기존 eval40 MSE에는 **새 UNet/VAE/text forward가 필요 없다.** 실제 생성은 새로운 noisy-latent trajectory라 A/B 캐시만으로 대신할 수 없다.

## 3. 가장 작은 1차 패키지
### A. 공개+사적 결합의 비보호 참고점을 먼저 계산
Static/full 각각 공개 환자와 사적 환자를 동일하게 가중하여, 결과 보기 전에 rho=80/112로 고정한다.

A_mix=(32*A_public+80*A_private)/112,
B_mix=(32*B_public+80*B_private)/112,
W_mix=(A_mix+.001I)^(-1)B_mix.

계산은 평균·작은 선형계 두 번이다. W_public, W_private, W_mix를 같은40명의 MSE로 비교하고, paired 환자 차이와 timestep별 차이를 기술한다. A/B만으로 전체 MSE는 즉시 가능하지만 timestep별 분해는 저장 raw/manifest를 다시 누적해야 한다. 이 분해를 첫 필수 범위로 늘릴 필요는 없다.

Rho를 eval40에서 검색하거나 private/public 비중을 결과가 좋아지는 쪽으로 바꾸지 않는다. 이 값은 새로운 “최적 비중” 주장이 아니라 동일 환자 가중치의 첫 참고점이다. 필요하면 향후 task가 공개/사적 target domain에 주는 비중을 따로 정의해야 한다.

### B. 후순위 조건부 대조: 공개 초기화 DP-SGD
기존 공개 선택 C, sigma, q=.1, T=2000, LR schedule, lambda=.001과16개 noise/sampling seed를 그대로 사용하고, W0만 공개 ridge로 바꾼다. Static/full×16=32회다. 기존 zero-init과 같은 randomness를 사용하면 초기화 효과를 직접 기술할 수 있다.

이 대조의 목적은 여전히 private80의 loss+lambda||W||²이다. **Public-initialized는 public+private objective와 같지 않다.** 오래 수렴하면 기존 private-only ridge로 가는 경향을 가진다. 작은 private-only 추가 이득 자체를 바꾸는 설계라고 포장하지 않는다. 같은 C/LR를 고정한 비교는 초기화 격리용이며 warm start에 최적화된 강한 SGD라는 주장도 아니다. 공개 초기화가 DP 비용을 늘리지는 않는다.

### C. 추가할 비교는 하나를 사전에 고르기
A의 결합 효용을 실제 보호 방법으로 시험하려면, 다음 두 계열을 혼동하지 않고 하나를 선언한다.

1. **원 결합 목적을 유지하는 public+private DP-SGD**를 권장하는 가장 직접적인 비교:
   업데이트의 데이터 항을
   2(1−rho)(A_public W−B_public)
   +rho*(Σ_private_selected clip_C[2(A_u W−B_u)] + sigma*C*Z)/(q*N0)
   로 두고, 밖에 2lambda W를 더한다. N0=80, W0=W_public. 공개 항은 clipping/noise 밖에서 계산하며 공개 자료를 고정한다. 사적 항과 Gaussian을 둘 다 rho배 하므로 기존 add/remove user 회계의 q,T,sigma를 유지할 수 있다. 이 진술은 feature bank와 public side information이 인접 데이터에서 고정되어 있다는 기존 조건에 의존한다. 새 helper/outer 및 독립 산술 검산이 필요하다.

2. **공개 reference의 표준 one-shot 대안**: 공개 Gram을 고정한 B 보호 또는 공개 W0의 한 번의 residual-moment 보정. 이미 저장 A/B로 가능하고 backbone 비용0이지만, 공개 covariance가 private covariance와 다르면 원 목적의 정확해가 아니다. 공개 중심 joint SSP 역시 가능하나 d² 좌표를 그대로 보호한다. 자세한 수식·absent/N0 경계는 [별도 메모](track1_public_reference_alternatives_20260916.md)에 있다.

최신 방향을 반영한 **권장 1차 묶음은 A와 §7의 실제 생성 경로 점검**이다. 공개+사적 결합 비보호 참고점과 기존 가중치를 이용해 캐시 MSE와 실제 sampling의 연결부터 확인한다. B의 public-init SGD와 C의 새 보호 변형은 후순위 조건부이다. 모든 centered variant·floor·rho를 한 번에 펼쳐 eval40에서 승자를 찾지 않는다. One-shot 효율이 논문의 주장이 되는 시점에는 공개 reference 대안이 직접 baseline이지만, 지금 모든 대안을 다 실행해야 A를 판단할 수 있다는 관문을 만들지는 않는다. Public Hessian/private gradient는 직접 선행이 있는 표준 비교이므로 새 연산의 novelty 근거로 삼지 않는다.

## 4. 무엇을 실질 이득으로 볼 것인가
기존 base 대비 약3%의 개선은 공개32만으로도 대부분 얻는다. 그 전체를 private utility라고 계산하지 않는다. Primary는 같은 head의 공개 baseline에 대한 추가 개선
G=(MSE_public−MSE_candidate)/MSE_public
이며, 생성/다운스트림 목표가 있어야 이 수치의 실용 의미가 생긴다.

- 매우 작은 paired 차이가40명 모두에서 같은 부호이거나 p-value가 작아도 실질 이득을 보장하지 않는다.
- 결합 비보호 참고점의 추가 이득 G_mix가 충분히 클 때만 DP가 그중 얼마나 유지하는지 계산한다. G_mix가0에 가까울 때 gain-retention 비율은 불안정하므로 사용하지 않는다.
- 현재 denoising MSE에 대해 검증된 임상·생성품질 최소 유효 차이는 없다. 따라서 “0.02%만 이겨도 성공” 같은 해석을 만들 수 없다.
- 연구 자원 배분용의 구체적 **예시**는 추가 MSE 상대개선 0.5%와 결합 참고 이득의50% 보존을 보조 진척 기준으로 사전에 선언하는 것이다. 이는 임상적으로 검증된 문턱이나 통계 정리가 아니며 이번의 채택된 자동 gate도 아니다. **MSE 미소 차이만으로 진행/중단을 결정하지 않고 실제 생성 경로의 효용·제약을 함께 확인한다.** 수치 문턱을 채택한다면 결과를 보기 전에 목표상 이유를 적고 결과가 안 나온 뒤 완화하지 않는다.
- Noise16개를 평균하고, 모델별·환자별 개선/악화와 분산을 전부 보고한다. 가장 좋은 noise seed의 모델만 생성하거나 비교하지 않는다. 16번은 데이터16개가 아니라 같은 데이터에서 메커니즘의 반복이다.

DP 모델의 공개 baseline 대비 이득이 있어도 head 제약·public access·epsilon8에서만의 결과다. 생성 품질과 개인정보 위험, 새로운 회귀법이라는 주장은 별개이다.

## 5. 평가 재사용과 tuning 오염
Eval40은 초기 MIA·capacity·DP 비교에서 이미 본 개발 자료다. 이번 추가 분석은 사후 진단이며 새로운 test가 아니다. C/scales/LR/T/PSD floor/rho/표현 선택을 이40명 결과로 반복 조정하면 그 결과로 일반화 증거를 만들 수 없다.

Public32도 이미 hyperparameter 선택에 사용됐다. 새 종류의 warm-start/pooled 메커니즘에 필요한 공개 calibration은 허용되지만 탐색 범위와 비용을 기록한다. Public32의16환자를5번 복제한 보정은80명의 독립 사적 분포를 재현하지 않는다.

잠긴 calibration/test 각140명은 이번 패키지의 가벼운 추가 점검에 쓰지 않는다. 후보와 실제 생성 품질 기준을 고정한 뒤 별도의 확인 단계에서, 프로젝트의 데이터 접근 범위와 비용을 명시하고 사용한다. 현재20–40명 효과가 약하다는 이유로 바로 잠금을 열어 유리한 숫자를 찾지 않는다. 이는 새 승인을 요구하는 규칙이 아니라 개발/확인 역할을 섞지 않기 위한 실행 설계이다.

## 6. 계산 시간과 작업 시간을 따로 적기
이미 측정된 값:
- Capacity의 private train/eval 특징3840회 추출: 314.687초(모델 로드·출력 포함).
- 공개512회 추출: 47.386초. 두 실행의 측정 forward는 warmup 포함4354회다.
- 공개 calibration 전체: 39.575초.
- 기존64 DP 모델+20control 실행·저장: 20.540초.
- SGD 한 회 평균: static .447초/full .534초. SSP 한 회 평균: .0010초/.0152초.
- 독립 private 검산: 24.970초. 이는 감사 작업 비용이며 모델 최적화 비용과 구분한다.

따라서 A의 새 solve는 일반적으로초 이내, B의32회 SGD는 현재 측정치 단순 합산 약16초 수준이다. C pooled32회를 추가하면 비슷한 수십초가 예상되지만 구현·공개 항 추가로 달라질 수 있고 아직 실측은 없다. 출력·hash·독립 검산까지 포함한 **순수 CPU 계산은 대략1–3분 범위의 계획치**다. Backbone 재추출0이므로 지금의 진단에 몇십 분 GPU 학습은 필요 없다.

반면 계약/목적 일치/paired seed/새 outer/독립 verifier/보고서가 필요하다. CPU만인 A+B는20–35분, C까지 포함하면35–60분 정도의 **사람/에이전트 구현·검토 작업 계획치**다. 우선순위로 바뀐 **A+생성 wrapper/검산/grid 묶음은 전체35–60분 정도**로 계획하는 편이 현실적이다. 기존 head generation 구현이 없으므로 “A/B solve가 빠르니 전체도1분”이라고 할 수 없다. 오류가 있으면 늘어나며 이 문서 자체가 실행/완료 약속은 아니다.

최종 효율 비교는 (a)공개 특징/보정 준비, (b)사적 특징 추출, (c)메커니즘 fit, (d)평가, (e)최종 generation 비용을 분리한다. 같은 cache를 쓰는 SGD도 backbone backward0이다. 기존 SSP의 millisecond solve를 SGD의 feature 추출 포함 시간과 비교하면 안 된다. 한 모델의 cold run과16번 연구 반복 비용도 구분한다. 공개 calibration을 amortize하면 양쪽에 같은 기준을 적용한다. **동일 품질을 달성하지 못한 방법에 “동일 utility에서 더 빠름”이라고 할 수 없다.**

## 7. 생성 샘플 점검은 별도의 작은 기능 검증
실제 generation에는 conv_out 입력 activation에서 동일 FP32 projection·시간 basis를 만들어 Wᵀphi를 기존 epsilon 출력에 더하는 wrapper가 필요하다. W=0 출력 exact, 기존 training packet에서 epsilon+head와 행렬 예측 일치, timestep basis/scheduler parameterization 일치부터 확인한다.

CFG는 조건부/무조건부 가지 모두의 head 적용을 미리 정의해야 한다. 현재 head는 weak-label 조건에서 학습했고 null caption dropout은 없다. Null branch에 자동 적용해도 학습 목적과 같다고 볼 수 없다. 첫 primary 시각 점검은 **guidance=1의 조건부 sampling**으로 단순화할 수 있지만 기존 CFG7.5 결과와 같은 조건의 비교라는 주장은 하지 않는다. CFG7.5는 별도 명시한 운용 대조로 둔다.

예비 작은 grid는 fixed4 prompts×2 seeds×4모델(base/public-only/pooled nonDP/기존 DP 1개)=32장, DDIM30이다. 기존 DP는 새 방법을 학습하지 않고 사전 고정한 표현/메커니즘의 repeat0을 사용하며 best seed를 고르지 않는다. 예를 들어 주표현full64와 기존 strong SGD 비교를 유지할 수 있다. CFG1이면960 UNet examples, CFG2-batch 계산이라면1920 examples다. 기존 .065초/단일 denoising example을 단순 적용하면 약1–2분 forward지만, sampling batch·VAE decode·로드·새 wrapper overhead 때문에 **실제 GPU wall-time은 미측정**이다. 구현·출력 일치 검산20–40분과 구분한다. 이 비용을 CPU 판별 계산 시간에 몰래 포함하지 않는다.

32장 grid는 artifact/도메인/조건 반응의 sanity이며 FID·진단 정확도·임상 안전성·개인정보 평가가 아니다. 구조가 보이지 않거나 심각한 artifact가 생기면 MSE만으로 확장하지 않는다. 반대로 예쁜 몇 장이 통계적 품질이나 privacy를 증명하지 않는다.

## 8. continue / revise / stop
- **Continue, 좁게:** A+생성 점검에서 결합 비보호 참고점의 사적 추가 정보가 실제 운용에 도움이 될 구체적인 여지를 보임. 그다음 공개 정보 활용이 공정한 DP 조건에서 얼마가 남는지 후순위 비교를 수행한다. 이미 계산한 DP 조건에서 작은 이득이 없다는 이유로 A+생성 점검을 막지 않는다. 후속 DP 이득은 seed·환자 편중과 독립 확인을 따로 검토하며 “논문 성립” 판정은 아니다.
- **Revise:** 결합 비보호 이득은 있지만 보호/초기화 후 사라짐. 공개 초기화, 같은 목적의 pooled SGD, 공개 Gram/reference 중 사전에 고른 표준 대안으로 clipping·noise·covariance 편향을 분리한다. 이 단계의 목적은 실패 원인의 범위를 좁히는 것이며 C/floor/rho를 private eval로 반복 검색하지 않는다.
- **Stop/hold 현재 축의 큰 확장:** 캐시 MSE와 실제 생성 경로를 함께 확인해도 이 제한 함수족에서 사적 추가 정보의 실질적 효용을 보여 줄 구체적인 근거가 남지 않거나, 생성 경로에서 심각한 artifact/조건 불일치가 확인됨. 이 경우 대규모 추가 DP 학습 투자를 멈추고 target domain/사적 자료의 추가 정보/표현을 재설계한다. **미소 MSE 차이 하나 또는32장 grid의 인상만으로 효용 부재를 입증했다고 하지 않는다.** DP diffusion 전체나 보호 효율 방향 전체가 불가능하다는 결론도 아니다.
- **미확인:**40개 개발 환자·한 projection·한 task에서 결과가 불안정하거나 최소 유효 차이 자체를 정의할 근거가 없음. 작은 p-value로 성공을 선언하지 않고 제한과 다음 비용을 명시한다.

## 9. 직접 읽은 범위와 근거
`run_patient_dp.py`, `dp_mechanisms.py`, `run_capacity.py`, `generate_quality.py`; capacity/public32/private-DP의 execution JSON; private-DP analysis/manifest 경로; 공개 params/public_only JSON; 두 patient_stats NPZ의 key/shape/split. 이번 NPZ 확인은 schema/역할만 읽었고 새 head solve나 결과 계산은 하지 않았다. 공개/사적 기존 결과의 원문을 위 숫자 근거로 사용했다. 기존 독립 검산은 public selection7,777항목, private41,812항목 PASS로 보존되어 있다.

이 메모는 최신 논문 전수 조사나 새 메커니즘 구현이 아니다. [공개 reference 대안의 수식/선행 메모](track1_public_reference_alternatives_20260916.md)와 연결되며, 이번에 허용된 새 산출물은 본 파일 하나다.
