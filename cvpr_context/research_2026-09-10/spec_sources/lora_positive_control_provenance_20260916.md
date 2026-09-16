# 기존 LoRA positive control의 출처·재현 조건
2026-09-16. 기존 report/PNG/checkpoint hash, loader/source, cohort 메타데이터를 읽었다. 두 PNG는 실제 view_image로 확인했다. 새 GPU·학습·생성·checkpoint 변경은 없다. 최종 calibration/test의 영상·latent·효용 결과는 열지 않았으며, 과거 M1 학습에 포함된 역할별 수는 manifest 메타데이터로만 집계했다.

## 1. 고정 대상과 실제 확인
대상은 기존 `model_1` step1000이다. 이번 선택은 lowest model name이며 기존 M2/여러 prompt에서 좋은 결과를 찾아 선택한 것이 아니다. 두 primary는 원 PROMPTS index1(no labeled finding), index2(pleural effusion)다.

공통 경로:
`code_working/_reports/cvpr_u_pilot_v1_001/quality/training_coverage_v2/step_1000_both/`

| 항목 | 고정 값 |
|---|---|
| report.json SHA | 670482cba7e7aa3261e6422fcb3ba8ba3afb011f96d7ea2d2319ad60e4bf41a7 |
| model_1_1.png SHA | dacf047603716883aac8c71a46398bf9baf4c1c62d841b034007dae6d61460b4 |
| model_1_2.png SHA | 15be12c84247bd0c7304761797a6cdef332d7f67f2a7809334e6247ba568a751 |
| M1 step1000 SHA | ae789c32ce9ebaca3eb568fda968c28d5b9cbdf39682d7a498615b26cbb08435 |

4개 SHA를 실제 파일에서 계산했다. 두 PNG와 checkpoint는 report의 값과 exact 일치한다. Checkpoint 경로는
`code_working/_reports/cvpr_u_pilot_v1_001/training_coverage_v2/model_1/step_1000.pt`이며 크기는20,261,578 bytes다.

두 PNG 모두 흑백 흉부 방사선영상의 전반적 구도가 보인다. 이는 이번 frozen residual preview의 줄무늬/분홍 물체와 구별되는 **gross domain positive-control 후보**라는 판단이다. 해부학적 정밀성·effusion의 의학적 존재·조건 정확성·임상 유용성을 판독하거나 인증하지 않았다. Report 자체의 상태는 여전히 `GENERATED_PENDING_VISUAL_REVIEW`, clinical_validation=false다.

## 2. 실제 seed와 prompt
| index | prompt | 저장 seed | 과거 기록 초 |
|---:|---|---:|---:|
| 1 | a frontal posteroanterior chest radiograph with no labeled finding | 535476699870967460 | 3.5484394999803044 |
| 2 | a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion | 594566474961504545 | 3.6081358999945223 |

원 코드의 seed 함수는 `common.SALT="cvpr-u-pilot-20260914-v1"`와 인자 `("quality-generation", index)`를 "|"로 결합한 SHA256 첫15hex의 정수다. CUDA Generator를 각 model/prompt마다 해당 seed로 다시 만들므로 원 코드상 base/M1/M2는 같은 prompt index에 같은 RNG 시작점을 사용한다. M1 checkpoint 초기화 seed260914와 generation seed는 서로 다른 값/역할이다.

## 3. 원 실행 경로
근거: `u_patient_audit/generate_quality.py:26–38`, `models.py:12–39`.

1. 같은 pinned SD2.1 snapshot revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`.
2. `load_unet(checkpoint,training=False)`: 원 UNet fp16 variant를 읽고 LoRA rank8/alpha8, to_q/to_k/to_v/to_out.0을 추가한다. 저장 adapter key 집합과 trainable key 집합을 확인해 checkpoint를 읽는다.
3. **Precision 세부:** 설치된 PEFT `cast_mixed_precision_params` 원문도 확인했다. 이 호출은 nontrainable base를 FP16, 당시 trainable LoRA를 FP32로 둔다. 이후 checkpoint 로딩·eval·freeze를 한다. 따라서 원 loader를 단순히 “모든 weight FP16”이라고 요약하면 부정확하다. 실제 계산은 아래 FP16 autocast 안에서 이루어진다.
4. `StableDiffusionPipeline.from_pretrained(...,unet=unet,torch_dtype=float16,variant="fp16",local_files_only=True)`. safety_checker/feature_extractor=None 및 requires_safety_checker=False는 원 실행과 같은 로딩 설정이다.
5. `DDIMScheduler.from_config(pipe.scheduler.config)`.
6. `torch.Generator(device="cuda").manual_seed(seed)`; inference_mode + CUDA autocast(float16).
7. Height=width=256, num_inference_steps=30, guidance_scale=7.5, output_type="np". Array를255배·clip 후 uint8 PNG로 저장했다. 원형은 후처리된 image만 저장한다.
8. Setup은 cudnn benchmark=False, CUDA/cuDNN TF32=False, deterministic algorithms=True이며 모델은 eval/frozen이다. enable_gradient_checkpointing flag는 켜지만 이 경로는 gradient 계산이 없다.

현재 읽은 source SHA:
- generate_quality.py: `eac6c6446c683c4bb50c929fc30982c6b8bfae0020bfcf30f4219a61357d8a0a`
- models.py: `cd1a0b0eca3f4460eadc247d62960eb50f60d09cc4c39c2ed279e5a3afb2fd91`
- common.py: `22ff06578ab7431ff9ac670eaef112371c800b13e9a3a4bf9cc3a5ba11b768d7`

Historical quality report는 generate_quality.py source SHA나 당시 torch/diffusers/PEFT/CUDA 버전·GPU 제품명을 직접 기록하지 않았다. 현재 source가 설명하는 경로와 report의 settings/이미지/checkpoint provenance를 구분한다. Training protocol은 models.py SHA를 결속하지만 generation report가 모든 실행환경을 동결했다는 주장은 못 한다.

## 4. Initial latent exact 재현의 한계
원 quality 폴더에는12개 PNG, comparison_grid.png, report.json만 있다. 원 생성 코드도 initial latent·trajectory·text embedding·scheduler timestep tensor를 저장하지 않는다. 따라서 **역사적 initial latent가 보관되어 있지 않다.**

Seed와 동일 CUDA RNG/라이브러리/dtype 호출 순서를 유지하면 재현을 시도할 수 있지만, seed만으로 당시 tensor가 bitwise 동일했다고 사전에 보장할 수 없다. Pipeline 내부의 latent 생성 device/dtype·추가 RNG 소비·VAE/attention 수치가 달라지면 PNG exact도 달라질 수 있다. 새 결과는 원 PNG의 exact 여부와 최대차이를 실제로 보고해야 한다.

원 PNG exact가 아니면 즉시 학습 효용 실패라는 뜻은 아니다. 같은 모델/조건을 사용했는지 수치 경로의 차이를 먼저 분리한다. 새 bridge에서는 initial Gaussian/실제 scaled latent·seed·dtype·DDIM timesteps·패키지 버전·모델/adapter fingerprint를 처음부터 저장하면 이 한계를 줄일 수 있다.

FP32로 전체 UNet을 바꾼 새 head witness와 FP16/autocast historical 생성은 같은 수치 경로가 아니다. 마찬가지로 원 CFG7.5와 새 조건부 CFG1은 다른 운용 설정이다. 둘을 몰래 섞어 parameterization/head 효과라고 부르지 않는다.

## 5. 훈련과 cohort 범위
`training_coverage_v2/protocol.json`에는 non_DP=true, rank8, AdamW lr1e-4/betas(.9,.999)/eps1e-8/weight_decay.01, batch4,1000steps,weak-label prompts가 명시되어 있다. 이 protocol 파일명이 contract.json은 아니라는 점도 확인했다.

저장 `model_1/report_1000.json`은 `PASS_TRAINING_EXECUTION_AND_COVERAGE`,912 training images 모두 실제노출,각4–5회,committed exposures4000,checkpointSHA일치를 기록한다. `verification_1000.json`도 checkpoint/exposure/optimizer/trace 무결성 PASS이며 “효용/프라이버시 인증 아님”을 명시한다.

M1 train manifest를 실제 집계한 결과:
| 과거 역할 | 환자 수 | E/train image 수 |
|---|---:|---:|
| fit | 40 | 80 |
| selection | 20 | 40 |
| calibration | 70 | 140 |
| test | 70 | 140 |
| background | 256 | 512 |
| 합계 | 456 | 912 |

현재 residual capacity의 train80 중40명, development40 중20명은 이 M1의 학습 환자다. Public calibration32와 M1 학습 환자 교집합은0이다. 이 역할은 이전 MIA 실험의 split 정의에서 온 것이며, 새 residual 연구의 독립 성능평가와 혼동하면 안 된다.

따라서 M1 positive control은 **같은 기반 모델의 기존 비보호 적응이 흉부영상 형태를 생성한 기록**을 재연결하는 용도다. 현재 head와 training patient수·목적·비용·표현력이 같지 않다. 이것을 환자DP baseline, 같은 자료에서의 공정한 품질 우위, 잠긴 환자의 독립 test 결과로 제시하지 않는다. 새 최종 환자 자료를 추가 학습하는 것은 아니지만, 과거 checkpoint가 이미 포함한 역할과 보호 부재를 감추지도 않는다.

## 6. 이번 감사의 결론
지정 M1/index1·2에는 검증 가능한 PNG/checkpoint/seed provenance가 있으며 실제 영상도 gross chest-domain 대조로 사용할 수 있다. 다만 historical initial latent·환경 전체는 저장되지 않았다. 먼저 동일 원 pipeline 경로를 확인하는 bounded bridge라는 목적이 적절하며, 이 두 장의 재현으로 질환 정확성·생성 품질 일반화·patient DP·새 head의 성공을 주장하지 않는다.

