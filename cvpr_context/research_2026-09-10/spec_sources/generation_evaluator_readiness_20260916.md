# 의료 생성 평가기 준비 수준 감사와 정정

2026-09-16. 코드·환경 메타데이터·모델 파일 SHA·프로토콜·실행/독립 검산 보고서·환자 명부만 확인했다. GPU, 새 encoder 출력·품질 점수, 학습은 실행하지 않았다. 기존 파일은 수정하지 않았다.

## 1. 정정할 결론

**RAD-DINO/BioViL-T 정량 평가 인프라는 이미 구현·실행·독립 검증되어 있다.** 앞선 재계획에서 “의료 생성 효용 평가기가 준비됐다고 확인하지 못했다”라고만 써서 이 준비물을 누락한 것은 불완전한 설명이었다. 정확한 상태는 다음과 같다.

> 의료영상의 특징분포·다양성·약한 텍스트 정렬을 비교할 평가기는 준비돼 있다. 독립 임상 타당성이나 downstream 의료 효용을 인증하는 평가기는 아니며, 현재 residual 생성물과 안전한 참조 명부를 연결한 실행은 아직 없다.

따라서 다음 패키지를 시각 grid만으로 제한할 이유는 없다. 기존 평가기를 개발용 정량 screen으로 재사용할 수 있다. 다만 기존 K5 계약과448명 명부를 그대로 가져오면 현재 연구의 환자 분리가 깨진다.

## 2. 준비 단계별 실제 증거

| 요소 | 구현 | 실행·검증 | 현재 residual에 재사용 |
|---|---|---|---|
| RAD-DINO encoder | exact revision110cbc18d5133582e320b43d53bf5c44e410c936, 고정 processor/로더 | 실제448장→[448,768], effective rank184.65758447, 첫8장 replay drift0. 독립 검산기가448장 전체 재인코딩해 일치 | encoder/전처리 재사용 가능. 새 생성 PNG 연결과 참조 명부 분리 필요 |
| BioViL-T image/text | revision692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23, hi-ml 이미지 모델 및 hash-pinned text 코드 | 실제448장→[448,128],7prompt→[7,128]. 전체 matched−deranged cosine .0605842215,95% [.0379566229,.0832236055]. 독립 전체 재실행 일치 | weak alignment screen 가능. 새 prompt 혼합에 과거 overall PASS가 자동 이전되지는 않음 |
| KID/PRDC | k5_evaluator.py의 unbiased cubic KID, deterministic subsets, PRDC k5, effective rank, descriptive Fréchet 구현. 전용6개 산술/재현 시험 기록 | 실제 encoder validity 실행은 real reference 대상. 그 gate에서 B0/M0/DP **생성물의 KID/PRDC 비교는 미실행** | 점수 연산 재사용 가능. 기존 KID subset50는 방법당16장에 적용 불가 |
| 생성물 평가 연결 | protocol에 P256 RGB 생성물을 BioViL용 grayscale로 바꾸는 정책 존재 | gate report는 generated_images_created=false. per-image features/scores도 미저장 | residual PNG manifest→고정 encoder→새 비교 보고서 연결 필요. 동결 K5 runner 수정 금지 |
| 임상 효용 | 연구 문서에 경계 명시 | 전문가 판독·진단 정확도·합성데이터 downstream 효용의 PASS가 아님 | 평가기 존재만으로 임상 효용 준비 완료라 정정하면 반대 방향의 과장 |

실제 상태는 PASS_K5_EVALUATOR_PREFLIGHT 및 PASS_K5_EVALUATOR_PREFLIGHT_INDEPENDENT다. 첫 독립 시도는 cuDNN TF32 누락으로 BioViL 평균이 약4.27e−5 달라 중단됐고, 설정 정정 후 전체 재실행 PASS 이력이 있다. 이번 감사는 저장된 보고서·소스·해시 확인이며 encoder 재실행이 아니다.

## 3. 현재 파일·환경 확인

ML 환경 code_working/base_gate/.venv/Scripts/python.exe에서 설치 메타데이터를 읽었다. torch2.6.0+cu124, torchvision0.21.0, transformers4.56.2, hi-ml-multimodal0.2.2, NumPy1.26.4, SciPy1.16.1, Pillow11.3.0으로 과거 보고서의 해당 버전과 일치했다. 현재 모델을 다시 로드·추론한 것은 아니다.

C:/Users/SOGANG/.cache/huggingface/hub의 아래 파일은 모두 존재하며 **실제 SHA256이 과거 public_report 값과 일치**했다.

- models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936/model.safetensors:346,345,912bytes.
- models--microsoft--BiomedVLP-BioViL-T/snapshots/692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23/model.safetensors:440,909,232bytes.
- 같은 BioViL snapshot의 biovil_t_image_model_proj_size_128.pt:109,745,561bytes.
- dp_training/k5_evaluator.py 및 run_k5_evaluator_preflight.py도 원 report의 source hash와 일치.

따라서 모델 재다운로드나 평가기 신규 연구가 선행조건은 아니다. 로더는 pinned local files를 사용하며 BioViL text의 알려진 unused BERT pooler 두 키 경계를 명시한다. 재사용 시 같은 TF32/precision/전처리와 source hash를 결속해야 한다.

## 4. 448명 명부의 실제 교집합

기존 real_evaluator_tasks_local.csv448명은 K5 public_development 참조다. 현재 CVPR 역할과 같다는 뜻이 아니다. 명부 ID만 교차 확인한 결과:

| 현재 집합 | 집합 환자 수 | 기존448과 같은 환자 | 동일 image ID |
|---|---:|---:|---:|
| fit | 80 | 32 | 22 |
| selection/development | 40 | 22 | 15 |
| 잠긴 calibration | 140 | **59** | 41 |
| 잠긴 test | 140 | **65** | 45 |
| 현재 head/clip 보정에 사용한 public quality | 32 | **11** | 8 |

**기존448 전체를 이번 생성 비교의 참조로 다시 계산하면 안 된다.** 잠긴 calibration/test를 열고, 공개전용 head가 본 환자를 독립 참조처럼 포함하는 문제가 있다. 이미지가 달라도 같은 환자이면 제외한다.

다섯 집합을 모두 환자 단위로 제외하면 **259명·259장**이 남는다. 조건별 잔여 수는 consolidation53, mass34, no-finding44, nodule53, effusion36, pneumonia23, pneumothorax16이다. 이 필터는 점수나 모델 결과를 보지 않고 정할 수 있다. 새 명세에 고정하되, 과거 encoder 사전검사에 이미 사용한 개발 참조라는 이력과 RAD-DINO의 NIH 사전학습 overlap을 보존한다. 새로운 완전 독립 임상 test라고 부르지 않는다.

과거 실행은 **per-image feature를 저장하지 않았다.** 따라서448명 캐시를 잘라 쓰는 방법은 없다. 허용된259명 또는 목적에 맞춰 사전 고정한 그 하위 명부만 새로 인코딩해야 한다. 이번 감사에서는 영상·점수를 새로 계산하지 않았다.

## 5. 방법당16장의 수치 계산과 한계

- Unbiased KID는 각집합 n≥2, PRDC k5는 각집합 n>5이면 코드가 동작한다. 방법당16장에서도 **수치 자체는 계산 가능**하다.
- 기존 K5 KID는 subset50×100회다. 이를 그대로 유지하려면 **방법당 최소50장**이 필요하다. generated16에서는 오류가 나므로 새 residual 명세에 subset≤16 또는 전체표본 unbiased KID를 따로 고정해야 한다. 기존 K5 코드는 수정하지 않는다.
- 여섯 방법의 생성물을 합쳐 n96으로 계산하면 모델 비교가 깨진다. 각방법16장의 paired 비교를 유지한다. 반복 KID subset은 새로운 생성표본이 아니며, generated16 전부를 매번 쓰면 반복변동은 주로 real subset에서 나온다.
- **DP head 혼합도 별도 문제다.** 기존 미실행 제안처럼16개 DP 가중치마다1장씩 만들어 합치면, 하나의 학습모델이 아니라16모델 혼합분포의 KID/PRDC다. 이를 개별 DP 모델 품질로 부르면 안 된다. 총96장을 유지하는 간단한 정정은 각 DP 방법에서 결과와 무관한 고정 manifest index0 한 개를 먼저 지정하고, 그 **같은 head**로16장을 생성하는 것이다. 이는 단일 DP 실행의 탐색 screen이며16회 DP 반복의 생성 품질을 평가한 것이 아니다. 후속 품질 실험은 각 고정 DP head에서 충분히 생성→head별 metric→DP seed 간 요약 순서여야 한다. 고정 head 정책을 채택하지 않으면 최초 정량 비교는 base/public/private/pooled 네 비DP 모델에만 한정하고 DP 그림은 연결 진단으로 남긴다.
- PRDC k5는16장 중 큰 비중이고 covariance/effective-rank도 표본 제약을 받는다. Unbiased KID가 작은 표본의 안정적 순위·높은 검정력을 보장하지 않는다. 유한 표본의 음수 KID를 오류로 잘라내지 않는다. **현96장은 연결·탐색 score이며 Gate1 private utility 통과가 아니다.**
- **Prompt/참조 혼합 정합 필요:** 기존7조건과 최근4prompt(generic/no-finding/effusion/cardiomegaly)는 다르다. Cardiomegaly는 과거7조건 validity 범위에 없고 generic은 질환 정렬의 정답조건이 아니다. 전체259명의7조건 분포를4prompt 생성물과 무심코 비교하면 조건 혼합 차이를 generator 열세로 오인할 수 있다. 실제 사용 목적에 따라 prompt/real-condition 가중치·해석 범위를 결과 전에 정한다. 과거 양수 alignment 조건만 골라 쓰지 않는다.
- BioViL 전체 gate는 양수였지만 consolidation/no-finding/effusion/pneumonia의 개별 matched−deranged 평균은 음수였다. 새4prompt 전체에 검증된 의료 alignment라고 소급할 수 없다. Descriptive cosine과 임상 정확도를 구별한다.

## 6. 다음 패키지의 재사용 방법과 시간

허용 참조 명부와 residual PNG manifest를 연결하는 **별도 얇은 평가 runner**에서 동결 core·encoder 로더 원리를 재사용한다. 원448 프로토콜의 실행·자동 gate는 호출하지 않는다. 모델별 KID/PRDC·다양성·BioViL weak alignment와 전체 이미지 grid를 함께 출력하면 시각검사만 하는 계획보다 객관적인 한정 근거가 생긴다. Clinical utility/최종 성능 주장은 별도다.

과거 RTX3070 실측은448 real RAD-DINO **30.693초**, BioViL image **11.388초**, text **.017초**, 전체 preflight **54.595초**다. Encoder peak는각약0.45GiB였다. 같은 전처리로259 real+96 generated=355장을 두 encoder에 넣는 연산은 이력상 **약1–2분 규모**, 로딩·입출력·metric을 포함한 실행 예산은 **5–15분**을 잠정 잡을 수 있다. **신규 경로 실측이 아닌 기존 report 기반 추정**이며 profile로 갱신한다.

Manifest/환자차단/prompt정합/작은표본정책/독립 산술 연결에는 추가 **30–60분 구현·검토**가 현실적이다. 기존 총2–4시간 패키지 안에서 검산/판독 시간을 재배분할 수 있으나 실제 범위가 늘면 **약2.5–5시간**으로 투명하게 조정한다. 대량 생성·다운스트림 임상 검증의 시간은 아니다. 이번 감사는 재사용 경로만 제안했고 실행하지 않았다.

## 출처

- [평가 core](../../../code_working/dp_training/k5_evaluator.py), [실제 preflight](../../../code_working/dp_training/run_k5_evaluator_preflight.py), [독립 verifier](../../../code_working/dp_training/verify_k5_evaluator_preflight_independent.py), [산술 시험](../../../code_working/dp_training/test_k5_evaluator.py).
- [동결 protocol](../../../code_working/_reports/nih_cxr14_k5_evaluator_preflight_protocol_v1_001/protocol.json), [real tasks448](../../../code_working/_reports/nih_cxr14_k5_evaluator_preflight_protocol_v1_001/real_evaluator_tasks_local.csv).
- [실제 report](../../../code_working/_reports/nih_cxr14_k5_evaluator_preflight_v1_001/public_report.json), [독립 PASS](../../../code_working/_reports/nih_cxr14_k5_evaluator_preflight_gate_v1_001/independent_verification.json), [원 gate 설명](../../../code_working/XRAY_K5_EVALUATOR_RESUME_PREFLIGHT_GATE.md).
- 교집합은 현재 [evaluation 명부](../../../code_working/_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv)와 [auxiliary 명부](../../../code_working/_reports/cvpr_u_pilot_v1_001/cohort/auxiliary_images.csv)의 patient/image ID만으로 계산했다. 임상 label이나 품질 점수로 환자를 선택하지 않았다.
