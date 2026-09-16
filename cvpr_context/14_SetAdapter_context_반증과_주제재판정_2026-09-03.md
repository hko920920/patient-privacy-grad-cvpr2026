# SetAdapter context 반증과 CVPR 주제 재판정

- 기록일: 2026-09-03
- 최신 판정: **표준 IID denoising + pooled patient context 방법군 종료**
- 이전 판정과의 관계: 13번 문서의 데이터 전제·구조 실행 가능성 PASS는 유지하지만,
  `PSPD가 현재 1순위 방법`이라는 추천과 `matched generation으로 진행`이라는 다음 행동은 이 문서가
  supersede한다.
- 다음 조건부 후보: **Patient-Private Longitudinal Residual Diffusion (PPLRD)**
- 중요 경계: PPLRD는 아직 방법 성공이 아니라 새 가설이다. 단순 longitudinal conditioning은 이미
  선행이 있으므로 신규성으로 주장하지 않는다.

## 1. 한 문장 결론

> 반복 영상을 무조건 함께 복원하게 하면 모델은 같은 환자의 다른 영상을 쓰지 않았지만, 앞선 합성
> 영상을 anatomy carrier로 두고 다음 시점의 변화 residual을 반드시 예측하게 하는 trajectory 생성은
> 환자 단위 DP와 결합할 별도 가설로 시험할 가치가 있다.

즉 PSPD의 **문제의식**—한 환자의 여러 영상을 하나의 privacy unit으로 보호해야 한다—은 남지만,
현재 SetAdapter **방법**은 반증됐다. 데이터에 same-patient 신호가 있다는 사실과, diffusion objective가
그 신호를 실제로 이용한다는 사실은 다르다.

## 2. 왜 generation 전체를 돌리기 전에 context gate를 했는가

13번까지 확인한 것은 두 가지뿐이었다.

1. frozen DINO 공간에서 같은 환자의 2--3개 X-ray가 다른 환자보다 구별된다.
2. Q=2 SD 2.1 mid block에 SetAdapter를 넣고 backward할 수 있다.

둘 다 `SetAdapter가 같은 환자 companion을 사용한다`는 증거가 아니다. 따라서 대규모 생성 학습 전에
primary X-ray의 timestep, corruption noise, prompt와 latent를 고정하고 companion만 같은 환자 대
다른 환자로 바꾸는 held-out causal control을 고정했다. 이 비교에서 이득이 없으면 set metric 향상도
generic capacity나 복제 collapse일 수 있으므로 full generation으로 가지 않도록 했다.

## 3. V1: 표준 SetAdapter held-out context-signal FAIL

### 동결 설계

- 이전 400 unique patients를 제외했다.
- exact Q=2에서 train 64 patients/128 images, validation 32 patients/64 images를 사용했다.
- frozen SD 2.1 base, no LoRA, 463,872-parameter SetAdapter만 192 AdamW steps 학습했다.
- 네 evaluation banks에서 primary input을 완전히 고정하고 `correct same-patient companion`,
  `finding-matched shuffled other-patient companion`, `adapter bypass`를 비교했다.
- primary gate는 `correct/shuffled <= 0.995`, correct wins `>=0.60`, target 0·1 각각 ratio `<1`,
  그리고 `correct/bypass <=0.995`였다.

### 결과

| 항목 | 결과 | 판정 |
|---|---:|---|
| correct / shuffled | `1.0002136004` | FAIL; correct가 미세하게 나쁨 |
| correct win fraction | `64/128 = 0.5000` | FAIL |
| target 0 ratio | `1.0001825810` | FAIL |
| target 1 ratio | `1.0002538042` | FAIL |
| correct / bypass | `0.9330780849` | PASS |
| primary-input max difference | `0.0` | integrity PASS |
| pre-training arm max difference | `0.0` | zero-init PASS |
| peak allocated CUDA | `1.86633 GiB` | resource PASS |

adapter는 bypass보다 평균 denoising loss를 약 6.7% 줄였지만, 올바른 환자 companion과 shuffled
companion을 전혀 구분하지 못했다. 따라서 이득은 patient linkage가 아니라 generic adapter capacity
또는 자기 record 경로에서 왔다. 실제 표준 block에는 `tokens + attended`가 있어 own-token residual을
통해 companion을 무시할 수 있었다.

- status: `FAIL_SETADAPTER_HELDOUT_CONTEXT_SIGNAL`
- protocol SHA-256:
  `C832321AF319FDEDA06669291AC0B165F321F487087A2277DFE6D55E86584B50`
- implementation SHA-256:
  `254F11864F7E9C7AB0395B8B77E2C67F80B9B9D42C4C5AE1832352BB8019EB5A`
- selection SHA-256:
  `B6951897C7270CC7A462496F82A5EC1CDDC65D7F37A15EA6E2D629809604E319`
- report SHA-256:
  `886C5E253877399C2311AD5C44A6761884BCD6D60BDC83D97E4A0B62B4624D06`

## 4. V2: strict cross-record-only 독립 확인도 FAIL

V1의 해석 가능한 구조 결함 하나만 고쳤다. diagonal/self value attention과 direct own-token residual을
모두 제거해 Q=2에서 record 1의 delta가 record 2의 value로만 만들어지게 했다. 이는 결과 전 동결한
유일한 mechanism 수정이며, cross-view attention 자체를 신규 기여로 주장하지 않았다.

### 독립성

- 64 V1 training patients와 동일 schedule/draws만 재사용해 architecture를 공정 비교했다.
- validation은 이전 400명과 V1 96명을 모두 제외한 뒤 새 32 patients/64 images를 뽑았다.
- 결과 전 protocol에서 실패 시 `standard IID denoising + pooled mid-block patient context` 전체를
  종료하고 attention mask, threshold, timestep으로 구제하지 않는다고 고정했다.

### 결과

| 항목 | 결과 | 판정 |
|---|---:|---|
| correct / shuffled | `1.0010702866` | FAIL; correct가 더 나쁨 |
| correct win fraction | `65/128 = 0.5078125` | FAIL |
| target 0 ratio | `1.0015495184` | FAIL |
| target 1 ratio | `1.0007231277` | FAIL |
| correct / bypass | `0.9393897207` | PASS |
| correct mean loss | `0.1708605924` | 기술통계 |
| shuffled mean loss | `0.1706779181` | 기술통계 |
| bypass mean loss | `0.1818846733` | 기술통계 |
| primary-input/pre-training difference | `0.0 / 0.0` | integrity PASS |
| permutation/singleton error | `0.0 / 0.0` | 구조 PASS |
| peak allocated CUDA | `1.86611 GiB` | resource PASS |

companion-only로 강제해도 `어떤 CXR companion`에서 얻는 population-level residual만 학습했고,
same-patient identity는 held-out denoising에 추가 이득을 주지 않았다. 두 버전이 서로 다른 fresh
validation에서 약 50% 승률과 ratio 1 이상을 냈으므로, full generation에서 우연히 set metric만
좋아지기를 기대하며 계속할 근거가 없다.

- status: `FAIL_CROSS_RECORD_CONTEXT_CONFIRMATION_CLOSE_STANDARD_IID_CONTEXT_CLASS`
- protocol SHA-256:
  `7EF39EB5F88220971EB9F5BAB99ADFB28C6CAB208ACDDAEA175D69B7E7491943`
- runner SHA-256:
  `0A4126A021BD0D77902FE224613F2C0D3F5220443B720EDF811B01A1279E5B73`
- cross-record adapter SHA-256:
  `51D8DE611F5D725D04FCC7B0FF46CA60CED5296881BCA01024320C128EC45783`
- selection SHA-256:
  `9145E378DAFD976ACAECEE6B78ED8F8B28D73BCDB13A466111DECBB648D038C0`
- report SHA-256:
  `68E8392A2B73CBA1030F24FD9AABFA0294D2EF12B2F41D806BA70F9C5B09081B`

## 5. 무엇이 정확히 종료됐고 무엇이 남았는가

### 종료

- standard IID noise/timestep에서 두 현재 records를 대칭적으로 함께 denoise하는 pooled SetAdapter
- own-token path만 제거하면 patient context가 살아날 것이라는 가설
- DINO same-patient retrieval PASS를 generator utility evidence로 쓰는 것
- correct/shuffled가 같은데 bypass 개선만으로 PSPD를 계속하는 것
- cross-record attention block 자체를 신규성으로 내세우는 것

SPAD, EpiDiff, MV-Adapter, CorrAdapter/CAMEO 등은 이미 cross-view interaction, correspondence와
consistency를 깊게 다룬다. 의료 적용이나 attention mask만으로는 독립 CVPR 기여가 되기 어렵다.

### 남음

- 반복 기록을 patient add/remove DP unit으로 잡아야 한다는 privacy 문제
- NIH에서 반복 환자 영상과 label 변화가 충분히 존재한다는 데이터 사실
- 기존 X-ray SD 2.1/LoRA, patient-DP clipping/accounting, 공격·평가 인프라
- unconditional symmetric context가 아니라 **이전 상태가 다음 상태를 정의상 조건화하는** 방향

## 6. 새 조건부 1순위: Patient-Private Longitudinal Residual Diffusion

### 한 문장

> 첫 합성 CXR을 만든 뒤, 이전 합성 CXR의 안정 구조를 carrier로 고정하고 경과 순서·임상 변화 조건에
> 따른 residual만 생성하는 transition diffusion을 환자 trajectory 전체 단위로 한 번 clipping/noising한다.

### 왜 이전 PSPD와 다른가

| 이전 PSPD | 새 PPLRD 가설 |
|---|---|
| 두 records를 대칭적으로 동시에 복원 | 이전 record에서 다음 record로 방향성이 있는 전이 |
| companion을 무시해도 denoising 가능 | prior 없이 다음 상태를 구성할 수 없도록 task를 정의 |
| set consistency가 주 평가 | stable anatomy 보존과 pathology change 정확도를 분리 평가 |
| pooled mid-block context | stop-gradient carrier + change-residual transition이 핵심 후보 |
| joint set 한 번 생성 | initial sample 뒤 synthetic trajectory를 순차 생성 |

훈련 중 real prior가 condition으로 들어가므로 그 record도 같은 환자 gradient 안에 포함하고, 전체
trajectory contribution을 하나의 patient vector로 만든 뒤 한 번만 clip/noise해야 한다. 배포 시에는
실제 환자 영상을 넣지 않고 첫 합성 영상부터 전 trajectory를 생성해야 `synthetic bundle release`가
된다. 실제 환자 prior를 inference input으로 받는 임상 forecast라면 training-data DP와 input privacy를
혼동하지 않고 별도 application으로 구분한다.

## 7. 근접연구 때문에 허용되는 신규성의 폭

| 직접 선행 | 이미 한 것 | PPLRD가 추가로 보여야 하는 것 |
|---|---|---|
| EHRXDiff, CHIL 2025 | previous CXR + subsequent EHR events로 future CXR 생성 | 단순 future-CXR claim 금지; patient-DP trajectory release와 DP-specific residual benefit |
| DDL-CXR, NeurIPS 2024 | previous CXR + irregular EHR로 individualized latent CXR 생성 | individualized conditioning claim 금지; 실제 input privacy와 training DP 구분 |
| longitudinal conditional diffusion, MICCAI 2024--25 | MRI/CT의 future state·missing scan·future embedding 예측 | time conditioning/residual 자체 금지; CXR patient-DP utility의 독립 근거 |
| P3SGD, CVPR 2019 | 한 환자의 모든 영상을 add/remove adjacency로 보호 | patient averaging/clipping 자체 금지; generative transition의 fixed-privacy 이득 |
| DPDM/DP-LoRA/medical DP-LDM | DP diffusion 및 private medical image generation | longitudinal patient unit, transition fidelity와 privacy attack을 함께 입증 |
| SPAD/EpiDiff/MV-Adapter/CorrAdapter/CAMEO | cross-view attention·correspondence·consistency | attention 자체 금지; temporally directed change mechanism이어야 함 |

따라서 PPLRD도 `first/최초`라고 쓰지 않는다. 중심 방법 claim은 **같은 patient-DP 예산에서 full-image
transition보다 stop-gradient anatomy carrier + change-residual parameterization이 변화 정확도와
identity 안정성을 함께 개선하는가**여야 한다.

## 8. 데이터 실행 가능성

로컬 NIH K10 `public_development`에는 4,831 images/1,816 patients가 있고, 886 patients가 두 장
이상이다. 로컬에 존재하는 records를 follow-up number로 정렬하면 3,015 adjacent-available pairs,
그중 exact `followup_no` gap 1은 2,279 pairs이며 65.14%에서 weak finding-label set이 바뀐다.

그러나 NIH metadata에는 실제 study date/time, report, medication/lab event가 없다. `Follow-up #`는
순서 index로만 사용하고 실제 경과시간 또는 임상 progression이라고 부르지 않는다. 로컬에는 현재
MIMIC-CXR 이미지·study timestamp·reports가 없다. 그러므로:

1. NIH는 order-proxy와 weak-label 기반 **가설 반증/개발**만 맡는다.
2. 최종 longitudinal paper는 MIMIC-CXR credentialed access 또는 동등한 timestamp+report 자료가
   필요하다.
3. NIH에서 성공해도 임상 시간 예측이나 치료 효과를 주장하지 않는다.

## 9. 갱신된 우선순위

1. **조건부 1순위: PPLRD** — 문제·기존 인프라·CVPR 시각 생성 범위는 잘 맞지만, NIH
   residual-premise와 MIMIC 접근성 두 gate를 통과해야 한다.
2. **안전 후보: patient-DP vs image-DP CXR generation benchmark/audit** — 이미 실행 기반이 가장
   강하지만 새 vision method가 약해 CVPR main-track 경쟁력은 1순위보다 낮다.
3. **고위험 후보: hierarchical patient latent + record residual joint generator** — PSPD 실패와
   multi-image diffusion 선행 밀집 때문에 현재는 보류한다.
4. **federated hospital extension** — PFDM과 직접 겹치고 multi-site 실험 부담이 커 첫 논문으로는
   우선하지 않는다.

이 순위는 PPLRD가 이미 가장 성공 가능하다는 뜻이 아니다. 현재 자료에서 **다음으로 가장 싸게
반증할 수 있고, 통과했을 때 남는 기여가 가장 명확한 후보**라는 뜻이다.

## 10. 다음 한 단계와 중단 규칙

다음은 생성 학습이 아니라 NIH order-proxy residual-premise 진단이다.

1. 이전 모든 528 patients를 제외한다.
2. exact consecutive `followup_no` pair를 환자당 하나만 선택하고 label-change/no-change를
   patient-disjoint train/validation으로 균형화한다.
3. frozen medical 및 generic feature 공간에서 `future = prior + predicted change residual`이
   `future = prior`와 label-transition derangement를 held-out 환자에서 이기는지 본다.
4. correct prior를 finding-matched other-patient prior로 바꾼 hard control도 함께 둔다.
5. residual이 copy baseline을 재현성 있게 이기지 못하면 NIH 기반 PPLRD를 중단한다.
6. 통과해도 MIMIC access 전에는 actual-time generation이나 patient-DP training을 시작하지 않는다.

## 11. 주요 1차 문헌

- CVPR 2026 PFDM:
  https://openaccess.thecvf.com/content/CVPR2026/html/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.html
- EHRXDiff, CHIL 2025:
  https://proceedings.mlr.press/v287/kyung25a.html
- DDL-CXR, NeurIPS 2024:
  https://papers.neurips.cc/paper_files/paper/2024/file/3310034c97fab48fdbcba18f90fd5364-Paper-Conference.pdf
- Conditional longitudinal radiological diffusion, MICCAI 2025:
  https://papers.miccai.org/miccai-2025/0164-Paper2656.html
- LoCI-DiffCom, MICCAI 2024:
  https://papers.miccai.org/miccai-2024/paper/1964_paper.pdf
- BrLP, MICCAI 2024:
  https://papers.miccai.org/miccai-2024/paper/0511_paper.pdf
- P3SGD, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_P3SGD_Patient_Privacy_Preserving_SGD_for_Regularizing_Deep_CNNs_in_CVPR_2019_paper.html
- SPAD, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kant_SPAD_Spatially_Aware_Multi-View_Diffusers_CVPR_2024_paper.html
- EpiDiff, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Huang_EpiDiff_Enhancing_Multi-View_Synthesis_via_Localized_Epipolar-Constrained_Diffusion_CVPR_2024_paper.html
- CAMEO, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Kwon_Correspondence-Attention_Alignment_for_Multi-View_Diffusion_Models_CVPR_2026_paper.html
- CorrAdapter, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html
