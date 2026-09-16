# PSPD 독립 데이터 확인과 Q=2 SetAdapter 구조 smoke

- 기록일: 2026-09-03
- 현재 판정: **data premise confirmation PASS + architecture smoke PASS**
- 허용 범위: **public matched non-DP falsification 설계로 진행 가능**
- 금지 범위: generation utility, patient-DP utility, privacy attack resistance 또는 CVPR-ready 주장

## 1. 결론

첫 4-record probe의 절대 cosine-gap 실패를 같은 자료에서 완화하지 않았다. 이전 240명을 모두
제외한 exact 2--3 record 환자 160명의 새 cohort에서 scale-free 기준 다섯 개를 결과 전에
동결했고 모두 통과했다. 이어 실제 SD 2.1 mid block에 463,872-parameter SetAdapter를 붙인 Q=2
forward/backward도 모든 구조 gate를 통과했다.

따라서 PSPD는 이제 단순 아이디어보다 한 단계 강하다. 공개 NIH 자료에는 low-multiplicity에서도
joint set modeling의 전제가 있고, 제안 구조는 현재 RTX 3070 자원에서 실제로 작동한다. 그러나
아직 **SetAdapter가 independent-generation baseline보다 좋은 patient set을 만든다는 증거는 없다.**

## 2. 완전 비중복 scale-free confirmation

### 고정 설계

- protocol SHA-256:
  `1ABE0C90DCF91D9FC83635358D1533381DA115994C68959976274C792240E212`
- clip calibration 144명, UCAN 16명, premise-v1 80명: 총 240명 제외
- NIH K10 public-development에서 정확히 2장 또는 3장인 환자만 사용
- `(target, record_count)` 네 cells마다 40명: 총 160 patients·400 images
- frozen DINOv2 ViT-B/14; embeddings 저장 없음; feature 실행 16.67초
- 절대 cosine 차이를 gate에서 제거하고 AUC, ordering, chance-normalized retrieval와 label variation을
  결과 전에 고정했다.

### 결과

| gate | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| pooled balanced same-patient AUC | `0.880605` | `>=0.75` | PASS |
| four-cell AUC | `0.863333--0.908472` | 각 `>=0.65` | PASS |
| patient ordering win rate | 전체 `0.975`, 각 cell `0.975` | 전체 `>=0.75`, 각 `>=0.60` | PASS |
| global R@1/chance | `0.48 / 119.7x` | `>=10x` | PASS |
| four-cell R@1/chance | `31.6x--39.17x` | 각 `>=5x` | PASS |
| multiple-label-set patients | `0.75625` | `>=0.30` | PASS |
| within-patient label Jaccard distance | `0.582240` | `>=0.10` | PASS |

네 strata의 방향이 모두 같아 target/control 또는 3-record 환자만의 효과로 설명되지 않는다.
within-patient이면서 exact finding label이 다른 203 pairs의 mean cosine `0.985816`은 같은 cell의
cross-patient same-label 6,970 pairs `0.972797`보다 높았다. 다만 DINO가 해부학 외에 촬영장치·자세
등 acquisition signature를 이용했을 가능성은 남으므로 clinical identity claim은 하지 않는다.

- status: `PASS_PATIENT_SET_DATA_PREMISE_CONFIRMATION`
- selection SHA-256:
  `E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8`
- report SHA-256:
  `45828BF811F3F7D198A11FBEB04F672DCEA6F9A2C6E9F99A92C95B671159F721`

## 3. 실제 SD 2.1 Q=2 SetAdapter 구조

### 최소 구조

1. 두 records를 SD 2.1 UNet이 동일 가중치로 처리한다.
2. mid-block `1280 x 4 x 4` feature를 record별 spatial mean으로 token화한다.
3. position embedding 없는 128-dim four-head self-attention이 같은 환자의 valid records만 본다.
4. shared FFN 뒤 token을 1,280-channel residual로 되돌린다.
5. output projection은 zero-init이고, padded slot은 attention key에서 제거하며, singleton은 exact
   identity branch를 탄다.
6. two-record denoising loss를 평균한 뒤 LoRA+SetAdapter 전체를 한 patient gradient로 보고 한 번
   clipping/noising하는 것이 향후 patient-DP 경계다.

SetAdapter는 463,872 fp32 scalars이고 기존 rank-8 LoRA 1,659,904와 합쳐 2,123,776개다. 전체
UNet spatial cross-attention을 추가하지 않아 Q=2 memory 증가를 제한했다.

### smoke 결과

| check | 결과 | 판정 |
|---|---:|---|
| pure selection/metric/adapter tests | `10/10` | PASS |
| integrated zero-init vs bypass max abs | `0.0` | PASS |
| Q=2 full backward | loss `0.155451`, missing/nonfinite grads `0/0` | PASS |
| LoRA gradient L2 | `0.136963` | PASS |
| SetAdapter gradient L2 | `0.000517903` | PASS |
| activated record-swap equivariance max abs | `0.0` | PASS |
| record 2 change -> record 1 output L2 | `0.0402214` | PASS |
| activated singleton vs bypass max abs | `0.0` | PASS |
| peak allocated CUDA | `1.81906 GiB < 7.5 GiB` | PASS |

zero-init 때문에 첫 backward에서 nonzero gradient는 output projection부터 생긴다. 이는 의도한
안정적 insertion이다. optimizer step 없이 output projection만 고정 random probe로 잠시 활성화해
순열 동등성과 실제 cross-record 연결을 확인한 뒤 모델을 폐기했다. DP noise, accountant,
checkpoint, latent/gradient/adapter 저장은 없었다.

- protocol SHA-256:
  `C6766B87010029DB412DBD45A62E9B4FB50E5CE275AFE717EC4B9E4DFCEB5DCD`
- implementation SHA-256:
  `19282445FA80C3BD8BF9BDFF37AE57A827848AA01E43B8DA67EC721B13AA8F42`
- report SHA-256:
  `71A799FF86BD2B485AACC50AF0F260BC1AAD3D9E1FED10EB5C36F022994EDF6E`

## 4. 이제 방법 가설은 무엇인가

핵심 대조는 다음이다.

> 동일한 patient-level DP adjacency, 동일 patient sampling/accounting, 동일 Q와 가능한 한 맞춘
> parameter·UNet-call 예산에서, 독립 record denoising보다 permutation-equivariant patient-set
> interaction이 set consistency를 real-patient 범위로 옮기면서 pathology diversity와 single-image
> fidelity를 유지하는가?

성공 조건은 단순 same-patient similarity 최대화가 아니다. 지나치게 높이면 같은 이미지를 복제하는
collapse가 된다. real patient set의 consistency distribution에 가까워지고, record 간 finding
variation, per-image fidelity, nearest-real-patient distance와 patient-level attacks가 함께 악화되지
않아야 한다.

## 5. 다음 실험과 중단 기준

다음은 바로 private training이 아니라 **public non-DP falsification**이다.

1. independent LoRA baseline과 SetAdapter arm을 같은 public patient-disjoint split에 둔다.
2. Q=2, 동일 optimizer steps·images·UNet calls를 맞춘다. adapter parameter 차이는 LoRA rank를
   줄인 exactly parameter-matched baseline과 원래 rank-8 capacity-matched baseline을 둘 다 둔다.
3. 먼저 작은 overfit sanity에서 SetAdapter가 cross-record signal을 학습할 수 있는지 확인한다.
4. 이어 bounded public training과 generation으로 FID/KID 또는 radiograph feature distance,
   pathology classifier, set-consistency distribution, LPIPS diversity, duplicate rate를 함께 본다.
5. set metric만 좋아지고 diversity가 줄거나, independent baseline 대비 반복 seed에서 안정적 이점이
   없으면 PSPD main-method를 중단한다.
6. 위 gate가 통과할 때만 기존 patient-DP trainer에 연결하고 새 LoRA+SetAdapter gradient norm으로
   clip calibration을 다시 한다. 기존 `C_patient`를 재사용하지 않는다.

## 6. 적용과 확장

- **주 응용:** 한 환자에 여러 흉부 X-ray가 존재하는 synthetic longitudinal/encounter bundle.
- **의료 확장:** ISIC 환자별 여러 lesion/view, mammography의 CC/MLO multi-view, retinal bilateral
  views. 각 데이터셋은 entity 정의와 set semantics를 별도로 고정해야 한다.
- **비의료 확장:** 동일 identity의 multi-view object/person sets. 의료 특화 주장을 일반화하려면
  별도 dataset과 privacy unit 정의가 필요하다.
- **federated 후속:** 병원은 client, 환자는 그 안의 nested privacy unit으로 둔 hierarchical PFDM.
  multi-site access와 시스템 복잡도가 커 첫 논문의 필수 구성에서는 제외한다.
- **AAAI audit 활용:** unit declaration, event trace, accountant·artifact binding은 supplement의
  실행 무결성에 재사용하되, CVPR 중심 기여는 set-valued vision method와 empirical result여야 한다.
