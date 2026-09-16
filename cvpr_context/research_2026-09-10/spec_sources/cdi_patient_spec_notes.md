# CDI 환자 확장 명세 독립 검토 — 2026-09-14

현재 위치는 2단계(가까운 방법의 적용 조건과 실패 원인 확인)이다. 이 메모는 원문·고정 소스·기존 산출물의 읽기 검토이며 새 모델 호출, 학습, 점수 계산, CDI 재현을 수행한 기록이 아니다. 기존 모듈·원점수는 변경하지 않았다. 2장이라는 이유로 CDI를 비교에서 제외하지 않는다.

검토 기준: X01 원문/보충자료, 공개 저장소 `code_sources/X01`(commit `dcd62258b0b3fde05d52aaecfade3b5f4c09507a`), 현재 cohort와 이미 측정된 SecMI/PFAMI/basic-controls 범위. 아래 파일 경로는 이 연구 폴더 기준이다. 보충자료 페이지는 `pdfs/X01_supp.pdf`의 PDF 물리 페이지이다. 전 문헌을 이번에 다시 읽었다는 뜻은 아니다.

## 1. 비교할 원형과 환자 확장을 구분한다

원형 CDI는 의심되는 공개 집합 P와 같은 분포의 미사용·미공개 참조 집합 U_ref의 여러 특징을 모아 logistic scorer를 학습하고, 별도 평가 부분의 점수 분포로 집합 사용 여부를 검정한다. 원문의 U_ref와 우리 연구의 U(같은 환자의 학습에 직접 들어가지 않은 영상)는 다른 개념이다. U 환자의 사진을 image-nonmember라는 이유로 patient-nonmember 참조로 쓰면 안 된다.

원문 §4.2–4.3(본문 pp.5–6), Appendix D/E(보충 PDF p.13)는 같은 분포의 reference와 control/test 분리를 요구한다. Appendix U(p.19)는 control 5,000 member + 5,000 nonmember에서 학습한 scorer를 별도 평가 자료에 적용하는 MIA 경로도 제시한다. 따라서 한 환자의 2장 안에 5fold를 억지로 만드는 대신, 별도의 fit 환자들에서 scorer를 학습하고 새 환자의 2장에 적용할 수 있다. 이는 타당한 patient adaptation 후보이며 아직 실행·검증된 결과는 아니다.

| 항목 | 원형/공개 구현 | 현재 환자 확장에서 명시할 것 |
|---|---|---|
| 단위 | 여러 이미지의 데이터 소유 집합 | 한 환자 2장; E와 U를 별도 평가 |
| fit 정보 | P_control/U_ref_control; benchmark에서는 train/validation 출처 | fit80의 target-specific 환자 참여 정답을 보조 정보로 허용하는지 명시 |
| 평가 | 분리된 집합 점수 + reference와 검정 | 환자 점수/AUC 우선; 원형 집합 검정과 별도 이름 |
| 분할 | 이미지 index의 balanced 5fold | 환자 단위 5fold; 같은 환자 E/U 및 두 모델의 레코드는 같은 fold |
| 특징 | white-box 26D | 동일 26D를 먼저 구현; 일부 특징만 쓰면 부분/gray-box 변형으로 표기 |
| 참조 | 같은 target으로 평가한 동일 분포의 미사용 자료 | 고정 base generator 자체가 reference 자료의 대체물은 아님 |

26D를 추출하고 보조 환자 fit에서 LR를 학습한 결과는 “CDI features를 이용한 patient-level MIA adaptation”으로 부를 수 있다. 원형의 reference 집합, 분리 검정, 추정 단위까지 재현하지 않았다면 “원형 CDI 전체 재현”이라고 쓰지 않는다. Appendix O(p.17)의 gray-box 변형처럼 GM/NO를 제외한 정당한 비교도 가능하되, 26D full white-box와 구분한다.

## 2. 정확한 26차원 순서

근거: `conf/attack/cdi.yaml`, `src/attacks/features_extraction/cdi.py`, `experiments/features_ablation.py:44`, `experiments/clf_shap.py:37`, `experiments/utils.py:32`. 아래 이름은 공개 코드의 표시 이름이며 index는 0부터이다.

| index | 원형 이름 | 내용 |
|---:|---|---|
| 0 | Denoising Loss | t=100, 독립 noise 5회 L2 평균 |
| 1 | SecMI$_{stat}$ | deterministic reconstruction L2 |
| 2 | PIA | t=200, p=5 |
| 3 | PIAN | 정규화한 PIA, t=200, p=5 |
| 4 | Gradient Masking_0 | t=0 |
| 5 | Gradient Masking_1 | t=100 |
| 6 | Gradient Masking_2 | t=200 |
| 7 | Gradient Masking_3 | t=300 |
| 8 | Gradient Masking_4 | t=400 |
| 9 | Gradient Masking_5 | t=500 |
| 10 | Gradient Masking_6 | t=600 |
| 11 | Gradient Masking_7 | t=700 |
| 12 | Gradient Masking_8 | t=800 |
| 13 | Gradient Masking_9 | t=900 |
| 14 | Multiple Loss_0 | t=0 |
| 15 | Multiple Loss_1 | t=100 |
| 16 | Multiple Loss_2 | t=200 |
| 17 | Multiple Loss_3 | t=300 |
| 18 | Multiple Loss_4 | t=400 |
| 19 | Multiple Loss_5 | t=500 |
| 20 | Multiple Loss_6 | t=600 |
| 21 | Multiple Loss_7 | t=700 |
| 22 | Multiple Loss_8 | t=800 |
| 23 | Multiple Loss_9 | t=900 |
| 24 | Noise Optimization_0 | 최종 평가 residual L2 |
| 25 | Noise Optimization_1 | 최적화 perturbation L2 |

주의: DL/ML은 MSE가 아닌 L2 norm이다. 기존 PFAMI original-loss MSE를 이름만 바꿔 넣을 수 없다. 동일 residual이라면 L2=sqrt(N×MSE)이지만 timestep, noise, repetitions까지 동일해야 원형 특징을 복원했다고 할 수 있다.

## 3. 데이터 비용과 실제 F/B 계약

F는 UNet의 이미지 1개 forward, B는 입력/perturbation 미분 backward로 센다. 배치 호출 1회를 이미지 1F로 축소하지 않는다. VAE, text encoding, CPU LR, reference 추출은 별도 비용이다.

| 모듈 | 원형 핵심 설정 | 이미지/target당 비용 |
|---|---|---:|
| DL | t100, 5 noise 반복 | 5F, 0B |
| SecMI-stat | 현재 source 정책 t0=0, t=100, step10 | 12F, 0B |
| PIA+PIAN | 초기 epsilon 1회와 두 종류 noisy input; 공동 추출 재사용 | 3F, 0B |
| GM | 10t, mask ratio .2; gradient + masked 평가 | 20F, 10B |
| ML | 10t, 하나의 noise를 10t에 공유 | 10F, 0B |
| NO | t100, L-BFGS-B maxiter5 | (q+1)F, qB |
| **26D 합계** | q=실제 NO objective 평가 횟수 | **(51+q)F, (10+q)B** |

`multiple_loss.yaml`의 n_repetitions=5는 실제 extractor에서 사용하지 않는다. 이를 보고 50F로 추산하면 틀린다. `noise_optim.py:55`의 maxiter5는 함수 평가 5회 보장이 아니다. line search를 포함한 nfev/njev/nit, 종료 상태, 실제 F/B를 저장해야 한다. 14D gray-box(DL/SecMI/PIA/PIAN/ML)는 공동 PIA 추출 기준 30F/0B이다.

fit80+selection40에 E/U 각2장, target2개를 새로 모두 측정하면 960 target-image records이다. 단순 산식은 48,960 + Σq F, 9,600 + Σq B이며 기존 캐시의 정확 일치 재사용분은 뺀다. reference64 환자의 2장까지 두 target으로 평가하면 별도 256 records가 더 필요하다. 이는 비용 명세이지 지금 실행 승인·확정 계획이 아니다. 원문 A100/64batch 시간표(Table8, 보충 PDF p.15)를 현재 SD LoRA 장비의 시간으로 옮기지 않는다.

필요 캐시:
- 각 record의 patient/image/model/role/실제 노출, checkpoint·latent·prompt·scheduler·code hash.
- 26D뿐 아니라 DL 5개 원손실, PIA/PIAN 중간 점수, GM 10개 점수와 mask/hash, ML 10개 점수.
- 각 모듈 noise seed와 실제 noise tensor/hash, t 순서·alpha·dtype.
- NO 최종 perturbation, 초기 latent/noise binding, objective trace와 optimizer 상태, 최종 residual 및 별도 최종평가 입력 hash.
- 입력 gradient 자체의 전체 영구 저장은 성능 비교에 필수는 아니나, 첫 conformance packet에는 GM gradient/mask와 NO gradient를 독립 검산할 수 있게 보존한다.
- 코드 원형은 npz 행 순서로 합칠 뿐 image_id join을 하지 않는다. 새 캐시는 ID로 join하고 중복/누락/모델 혼합을 차단한다.

## 4. 원문·공개 코드의 실행 지뢰

### 4.1 GM 입력 차이

`gradient_masking.py:9`는 noised latent z_t에서 residual L2의 입력 gradient를 구하고 absolute top-k 20%를 mask로 만든다. 그러나 `:47`의 noise_top_values는 **clean latent를 clone하여** 해당 좌표를 noise로 교체한다. `:68` 최종 residual은 mask에서 (noise-clean_latent)-prediction이다. 원문이 설명하는 noised z_t의 해당 좌표 교체와 literal 코드 경로를 혼동하면 안 된다. 우선 공개 코드 그대로의 수치 경로를 확인하고, 수식 의도에 맞춘 변경은 명시적인 별도 variant로 기록한다.

GM의 미분 대상은 입력 latent다. target weight를 고정하고 입력 gradient만 허용해야 하며, 일반 inference_mode로 전체 함수를 감싸면 잘못된다. absolute top-k tie, FP32/FP16, batch 평균에 따른 gradient scale도 기록한다.

### 4.2 NO 최적화와 최종 평가 차이

`noise_optim.py:11`의 objective는 z_t+δ를 이미 noised 입력으로 넣는다(noise argument 없음). 반면 `:74`의 최종 평가는 z_t+δ와 원 noise를 함께 전달한다. `GeneralLatentDiffusionWrapper.py:58–85`는 noise가 있으면 noise_latents를 다시 호출한다. 따라서 literal release의 최종 residual은 최적화한 objective를 그대로 재평가한 값이 아니다.

epsilon 모델의 q_sample 표기라면 objective 입력은 z_t+δ, literal final 입력은 sqrt(alpha_t)(z_t+δ)+sqrt(1-alpha_t)epsilon이다. 이 차이를 무시하고 최종 호출에서 noise를 빼면 “원형 그대로”가 아니다. 반대로 literal 경로를 실행하고 최종 score가 최적화 objective라고 설명해도 틀린다. NO perturbation 전체 latent 차원(현재 4×32×32=4096)을 쓰며, 기존 R의 저차원 조건 계수 최적화로 대체하면 NO 재현이 아니다. batch NO는 하나의 평균 objective/line-search 종료를 공유하므로 첫 검사와 환자 고유 점수는 batch1로 고정하고 그 차이를 명시하는 편이 재현성이 좋다.

### 4.3 alpha/time/encoding/condition

`CompVisLatentDiffusionWrapper.py:26`는 실제 모델 q_sample, `:46`는 모델 alphas_cumprod[t]를 쓴다. `:33`은 class_label conditioning과 apply_model 출력이다. 현재 backend의 generic text hidden, cached latent mode/scaling, scheduler alpha, timestep 정수 인덱스와 동일하다고 가정하면 안 된다. 학습에 쓰인 현재 scheduler 전체 alpha와 prediction_type을 계약에 묶는다. epsilon이면 직접 사용하며 v_prediction이면 올바른 epsilon 변환을 거친다는 점도 별도 검산한다. t=0은 alpha=1/무노이즈라는 가정을 하지 않는다.

General wrapper의 기본 encode/alpha/predict/noise 함수는 random/placeholder이다. **base wrapper 객체를 그대로 실행하는 것은 실제 원형 검산이 아니다.** 재사용 가능한 것은 predict_noise_from_latent의 noise/use_grad 분기이며, encode/noise_latents/_predict_noise/get_alpha는 실제 현재 backend로 모두 구현해야 한다. CompVis의 class conditioning을 current text conditioning에 연결한 것은 모델 backend adaptation이다.

### 4.4 scorer 경로의 실제 오류와 수학 보존 수정

`src/attacks/scores_computation/cdi.py:31`은 numpy를 import하지 않고 np.arange를 쓴다. 같은 파일 `:19–24`는 train_dataset["data"]/["label"]을 사용하지만, `utils.py`의 MIDataset.__getitem__은 숫자 index로 tensor를 조회한다. 적어도 해당 경로는 수정 없는 정상 실행을 가정할 수 없다. feature assembly의 B×1×26을 Nx26으로 펼치는 것도 확인해야 한다.

작동 의도가 분명한 비교 기준은 `experiments/cdi_perf.py:51–90`이다. 여기서는 .data/.label, train-only StandardScaler, LR(solver=liblinear, random_state=0, max_iter=1000)을 사용한다. config 경로 기본 solver와 같다고 쓰지 않는다. 새 adapter에서 .data/.label로 올바르게 조회하고 Nx26으로 구성하는 것은 명시적 인터페이스 수정이며 LR 수학을 변경할 필요는 없다. 원본 동결 파일을 고치거나 main을 실행하지 않는다.

fold score를 concat하면 원래 patient/image 순서가 사라질 수 있다. 반드시 held-out index로 복원한다. 원형 동일 index를 P/U 양쪽에 적용하는 5fold는 balanced count를 가정하므로 환자 그룹 stratification으로 교체 사실을 기록한다.

### 4.5 정규화·방향·상관·clipping·통계

- source LR는 학습 fold에서만 StandardScaler를 fit한다. selection/cal/test와 reference-test에 fit_transform하면 누출이다.
- `utils.py:12`의 member=1/nonmember=0, predict_proba[:,1]이 high-member다. 원손실 26D는 positive norm으로 유지하고 LR가 방향을 학습한다. 기존 -SecMI를 원형 feature1에 넣는다면 L2로 되돌리거나 변환을 명시한다.
- 검토한 26D assembly/working LR 경로에는 feature clipping, PCA, decorrelation이 없다. 이것을 원형 구성요소라고 적지 않는다. 추가하면 별도 사전 고정·fit-only ablation이다.
- 상관된 여러 loss를 넣었다고 독립 증거 26개가 되는 것은 아니다. LR의 결합 이득과 환자 관계 모듈의 추가 이득을 구분해야 한다.
- `src/evaluation/evaluate.py:19`의 joint-max scaling은 signed score 최대값이 음수/0일 때 방향·정의 문제가 생길 수 있다. 신규 LR 확률과 환자 AUC는 명시적 high-member 방향으로 직접 계산한다.
- 원문은 one-sided Welch test를 설명하지만 `evaluate.py:89`는 SciPy 기본 two-sided ttest_ind(equal_var=False)와 별도 is_correct_order를 반환한다. 어느 계약을 따르는지 명시하고 p를 환자 membership probability라고 부르지 않는다.
- 원문의 여러 owner-draw p 평균(Appendix E)은 한 환자 2장으로 1000개의 독립 환자 증거를 만드는 절차가 아니다.
- `evaluate.py:76`의 tpr[np.sum(fpr < threshold)]는 요구 FPR을 넘는 ROC 지점을 고를 수 있다. 환자 20 nonmember의 selection에서 1% FPR 실용성을 주장하지 않는다.

## 5. 현재 split과 fit label 허용 범위

현재 평가 환자 400명은 fit80 / selection40 / calibration140 / test140이며 각 split의 A/B가 균형이다. 모델1은 A, 모델2는 B를 학습한다. 둘은 여러 환자가 교환되고 같은 초기 seed·배경을 쓰므로 독립 seed 반복이나 단일 환자 DP neighbor가 아니다. 모든 target별 선언 label은 실제 training exposure와 대조해야 한다.

fit80에서 A40/B40을 5fold로 나누면 각 validation fold에 A8/B8, 총16명의 환자를 둘 수 있다. E/U 각2장을 같은 환자 fold에 묶고 두 target의 fold도 공유한다. 모델별 scaler/LR는 별도로 fit하며, 두 모델 점수를 동시에 아는 attack을 묵시적으로 만들지 않는다.

보조 membership 정답을 쓰는 LR는 이 공격 접근권을 명시하고 비교군에도 같은 fit80 정보·튜닝 기회를 준다. 실제 환자 스스로는 이런 정답 80명을 갖지 않을 수 있으므로 label-free patient-side threat model과 구분한다. 가능 경로는 다음과 같으며 어느 하나를 결과 보고 후 유리하게 고르지 않는다.

1. fit E 이미지의 exact image label로 image scorer를 학습 → E/U 환자2장 mean score: U로의 전이를 보는 경로.
2. fit U 이미지에 patient participation label을 붙여 scorer를 학습 → held-out U 환자: patient-labelled auxiliary control을 허용한 강한 경로.
3. 이미지 26D를 환자 평균 26D 또는 평균+최대 52D로 결합한 LR: 표준 환자 결합 baseline이며 원형 image scorer와 다른 adaptation.

m=2에서 top-1=max다. mean/max를 쓴다는 사실만으로 새로운 구조 기여는 아니다. 같은 환자 E를 fit하고 U를 validation에 두는 것은 patient-wise 검증이 아니다. calibration/test는 이번 명세와 개발 fit에 열지 않는다.

## 6. 기존 캐시만으로 가능한 범위

| 기존 산출물 | 실제 측정 범위 | CDI 재사용 한계 |
|---|---|---|
| latent/prompt cache | 전체 잠금 cohort | 입력은 재사용 가능; 새로운 GM/NO 값을 포함하지 않음 |
| SecMI secmi_v1 | fit 원래8명 + selection40, E/U 각2장, 두 target | feature1 재사용 후보. ID·정밀도·설정이 정확히 같아야 함 |
| basic_controls | fit 원래8명; E/U, generic/matched; 5t×2draw | 26D DL/ML와 t/noise/repetition이 다름 |
| PFAMI pfami_v1 | selection40 전체; benchmark는 fit1명 | relative/difference/original-loss는 CDI의 GM/NO/PIA 캐시가 아님 |
| public encoder | fit80 등 고정 cohort | target-independent 비교로 유용하나 CDI feature로 부르지 않음 |

GM/NO와 PIA/PIAN의 완성 캐시는 없고 fit80 전체의 공격 특징도 없다. 따라서 현재 CPU만으로 fit80에서 full CDI를 학습해 selection40을 채점할 수 없다.

정확히 가능한 제한적 CPU 작업:
- SecMI와 encoder 특징: 공통 index에서 사전 고정 결합을 검토할 수 있으나 CDI가 아니다.
- fit8의 SecMI+basic fixed-loss: 같은 fit8 안에서 환자 단위 exploratory CV가 가능하다. 표본이 작고 selection40 검증과 다르다.
- selection40의 SecMI+PFAMI 특징: 이미 관측된 개발 자료에서 patient-wise cross-fit 결합을 할 수 있으나 새 test가 아니다. fold 밖의 scaler/fit/feature selection을 금지하고 development-only로 표기한다.

**중요한 교집합 제한:** fit8 basic loss는 [50,250,500,750,950]×2draw이고 PFAMI selection original loss는 [0,50,...450]×1draw다. fit8 basic loss로 학습하고 selection PFAMI loss를 같은 feature인 양 채점하면 안 된다. PFAMI fit1은 한 target에서 한 class만 있으므로 그것만으로 정상 supervised member/nonmember LR를 fit할 수도 없다. 앞서 “fit8의 기존 점수 CPU 결합 가능”이라는 설명은 fit8 내부 분석 범위이며 fit80-fit/selection40-eval 완성 경로라는 뜻이 아니다.

## 7. 가장 먼저 구체화할 실행 단위와 conformance 전략

다음 실행 후보는 **기존 fit 환자 1명에서 26D extractor의 경로·비용을 확인하는 kernel benchmark**다. 이것은 성능/누출 양성 대조나 CDI 환자 확장 완료가 아니다. 먼저 한 target·한 이미지로 literal source path를 확인하고 필요 시 같은 환자의 E/U 및 두 target까지 고정 범위로 확장한다. 이 메모에서는 어느 실행도 하지 않았다.

API 제안:
- source_bindings(): pinned commit, 원형 extractor/config/wrapper SHA.
- extract26(backend, latent, conditioning, per_module_noise, policy): features[26], module detail, F/B, seconds, optimizer trace.
- backend 계약: encode(고정 latent 반환), noise_latents(잠긴 alpha의 q_sample), predict_noise_from_latent(원형 noise/use_grad 분기), get_alpha_cumprod, _predict_noise(현재 epsilon backend).
- target weights는 고정하고 입력미분만 허용. norm은 원형대로, FP32 batch1, module별 noise를 명시하며 model/patient 비교에서 재사용 정책을 잠근다.

원형 process_batch/obj/get_masks/get_final_loss를 그대로 호출하는 얇은 adapter를 우선한다. 불필요한 원형 main import 의존성을 피하려면 실제 source AST에서 필요한 class/method를 읽어 compile하되 source hash와 정확 추출 노드를 기록한다. 소스상 버그/placeholder를 몰래 실행하지 않으며 CompVis 모델 전체를 로드했다는 주장도 하지 않는다.

첫 CPU conformance는 두 층이다.
1. 동일 alpha/latent/noise와 미분 가능한 작은 결정적 epsilon 함수를 넣어 원형 wrapper+extractor와 독립 수식 구현의 loss, gradient, GM mask, NO final double-noise를 비교한다. 이것은 backend 접합과 산술의 synthetic 검사다.
2. 실제 현재 target의 첫 이미지 packet에서는 latent/noise/alpha, 예측 residual, GM 입력 gradient/mask, NO objective/최종 입력/perturbation을 보존하여 독립 검산한다. 실제 target derivative의 완전한 재현은 target 호출 또는 필요한 Jacobian 정보 없이 CPU residual만으로 증명되지 않는다. 저장 packet 검산과 독립 GPU forward 검산을 혼동하지 않는다.

현재 pinned X01 트리에서 normal-operation 정답 feature/gradient를 담은 공개 CPU replay packet이나 conformance test fixture는 찾지 못했다. README:45–55는 모델/데이터 다운로드를, :102–109는 직접 feature extraction을 안내한다. 이 범위 밖의 원격 저장소에 packet이 전혀 없다는 확인은 아니다. 그러므로 synthetic 검사를 “공개 원형 정답과 일치”라고 표현하지 않는다.

예상 구현 시간(진행 전 추정): 얇은 26D source adapter와 synthetic 수치 경로 확인에 약 15–25분, runner/protocol/독립 검산은 병렬 준비 가능하다. GM/NO source divergence나 scipy dtype 문제가 실제로 드러나면 이 범위를 넘길 수 있으며 먼저 실제 경과와 원인을 보고한다. GPU 시간은 q와 현재 input-gradient backward를 아직 측정하지 않아 수치 확정하지 않는다. 한 장의 kernel 결과로 full fit80/selection40 시간과 시행 여부를 구체화한다.

## 판단 범위

현재의 결론은 “CDI가 실패했다”가 아니라 **실제 환자 확장이 아직 구현·평가되지 않았다**이다. 원형 공개 구현의 문제를 고쳐 수학을 보존하는 wrapper, 동일 reference/보조 정보의 공정성, patient-wise 분리, 실제 26D 캐시가 우선이다. 기존 SecMI/PFAMI 결합을 먼저 탐색할 수 있으나 full CDI를 대체했다거나 기존 연구의 한계를 입증했다고 기록하면 안 된다.
