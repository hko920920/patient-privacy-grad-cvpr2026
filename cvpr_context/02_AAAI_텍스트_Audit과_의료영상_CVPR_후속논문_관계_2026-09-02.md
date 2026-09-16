# AAAI 텍스트 Audit과 의료영상 CVPR 후속논문의 관계

- 기록일: 2026-09-02
- 성격: 연구 방향 및 투고전략 기록
- 상태: 제안 단계. 현재 동결 실험계약이나 활성 학위논문 원고를 변경하지 않음

## 한 문장 결론

**이 주제는 CVPR의 범위에는 분명히 맞지만, AAAI audit을 의료영상에 그대로 적용하는
도메인 확장이 아니라 환자 단위로 기여를 집계한 뒤 clipping/noising하는 diffusion-specific
patient-DP 방법과 image-DP checkpoint의 재사용 가능 경계를 새 기여로 세울 때 CVPR main
paper로서 가장 설득력이 높다.**

즉, `CVPR에 맞는 주제인가`와 `현재 형태 그대로 CVPR에서 경쟁력이 있는가`는 구분한다.

- 주제 적합성: 높음
- 현재 audit+비교 설계만의 경쟁력: 중간
- patient-aware diffusion 방법, 강한 공격·효용 평가, 복수 데이터셋까지 갖춘 경우: CVPR
  main을 노릴 만함
- 이는 적합성 판단이지 채택 가능성 보장이 아님

## 1. 현재 AAAI 연구의 위치

사용자가 설명한 현재 상태는 다음과 같다.

- AAAI용 약 7쪽 compressed audit 논문이다.
- 중심 데이터는 텍스트/비영상 계열이다.
- 본문과 appendix를 포함한 논문 구성을 이미 완성했다.
- 평가 결과나 채택 여부는 아직 확신할 수 없다.

확인된 로컬 후보 산출물은 다음과 같다.

- `../포함 SCI 연구들/main_audit_aaai27_v24_candidate.pdf`
- `../포함 SCI 연구들/main_audit_aaai27_supplement_v15_candidate.pdf`
- `code_originals/paper_sources/auditable_privacy_v24_v15/`
- `code_originals/_archives/audit_code_and_data_supplement_aaai27_v17.zip`

위 기록은 파일명과 사용자 설명에 근거한 현재 작업 상태다. 이 문서는 해당 논문의 실제
투고·심사·채택 상태를 별도로 단정하지 않는다.

## 2. 두 논문의 역할 분리

| 구분 | AAAI audit 논문 | 제안 의료영상 논문 |
|---|---|---|
| 중심 질문 | 공개된 DP 주장과 실행 증거가 privacy unit, accounting, composition 관점에서 일치하는가 | 복수 영상을 가진 환자에서 image-DP 근거가 patient-level 배포 주장을 언제 허용하며, 언제 patient-DP 재학습이 필요한가 |
| 주된 도메인 | 텍스트/비영상 데이터와 일반 audit 사례 | 의료영상 latent diffusion |
| 주된 기여 | 사후 검증 체계, schema, verifier, evidence lineage | patient-aware private diffusion 학습, image-to-patient accounting 경계, 환자 단위 공격·효용·비용 비교 |
| 산출물 | privacy claim의 audit 결과와 검증 가능한 증거 | 재사용 가능/불가능/재학습 필요를 판정하는 release decision과 생성모델 실증 |
| 관계 | 기반 audit 인프라 | audit을 한 구성요소로 재사용하되 별도 과학 질문과 방법을 갖는 후속 연구 |

따라서 의료영상 연구는 `같은 audit을 이미지에 한 번 더 실행`하는 논문이 아니다. AAAI의
audit 체계는 측정·검증 모듈이고, 새 논문의 본체는 **privacy unit mismatch 때문에 생기는
diffusion 학습 및 배포 문제**다.

## 3. AAAI 결과에 따른 세 가지 분기

### 3.1 AAAI가 채택되는 경우

- AAAI 연구를 선행연구로 명확히 인용하고 audit schema와 verifier를 재사용한다.
- CVPR 원고에서는 일반 audit 체계를 다시 신규 기여로 주장하지 않는다.
- 새 기여를 patient-aware diffusion optimization, patient-level accounting, 환자 단위
  memorization/MIA, 의료영상 utility와 reuse-versus-retrain frontier에 둔다.
- 가장 자연스러운 관계는 `기존 audit framework의 단순 domain extension`이 아니라
  `기존 framework로 발견한 문제를 해결하는 vision-specific method paper`다.

### 3.2 AAAI가 탈락하는 경우

- reviewer feedback을 audit 모듈과 보고 방식 개선에 사용한다.
- 의료영상으로 데이터만 바꿔 같은 기여를 재포장하지 않는다.
- AAAI 채택 여부와 무관하게 의료영상 논문 하나만 읽어도 문제, 방법, 실험, 결론이 완결되게
  만든다.
- 탈락은 의료영상 방향의 근거를 없애지 않는다. 다만 audit 자체가 이미 충분히 강하다는
  전제도 두지 않는다.

### 3.3 AAAI가 심사 중인 경우

- 설계, 코드 골격, 데이터 split, 소규모 pilot까지 준비할 수 있다.
- 동시 투고 시에는 목표 연도의 CVPR dual-submission 및 substantially-similar 기준을 다시
  확인한다.
- 두 원고의 contribution paragraph, 표·그림, 결과와 문장 중복을 제출 전에 정량·정성 모두
  점검한다.

## 4. 왜 CVPR 범위에는 맞는가

2026년 공식 Call for Papers는 다음 영역을 명시한다.

- image and video synthesis and generation
- medical and biological vision, cell microscopy
- transparency, fairness, accountability, privacy and ethics in vision
- datasets and evaluation, vision applications and systems

제안 주제는 **의료영상 + diffusion generation + privacy + evaluation**의 교차점이므로 scope
fit은 높다. 또한 공식 reviewer training은 방법론적 기여뿐 아니라 의미 있는 empirical 또는
conceptual contribution도 평가 대상으로 설명한다. 다만 기술적으로 건전하고 CVPR 독자에게
중요하며 충분한 파급력을 가져야 한다.

판단에 사용한 공식 자료:

- CVPR 2026 Call for Papers: <https://cvpr.thecvf.com/Conferences/2026/CallForPapers>
- CVPR 2026 Reviewer Training Material:
  <https://cvpr.thecvf.com/Conferences/2026/ReviewerTrainingMaterial>
- CVPR 2026 Reviewer Guidelines:
  <https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines>
- CVPR 2026 Author Guidelines:
  <https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines>

현재 확인 가능한 공식 근거는 CVPR 2026 기준이다. 실제 목표가 CVPR 2027 이후라면 해당
연도의 CFP, page limit, concurrent/dual-submission 규정을 제출 전에 다시 확인한다.

## 5. 왜 단순 도메인 확장만으로는 부족한가

다음 형태만으로는 CVPR main에서 incremental/application paper로 평가될 위험이 높다.

1. 기존 DP-LoRA 또는 기존 user-level DP를 ISIC에 그대로 적용
2. 기존 audit 도구를 그대로 실행
3. image-level과 patient-level 숫자 차이만 표로 비교
4. 일반 생성 품질 지표만 보고

이 구성도 유용한 실증 연구가 될 수 있으나, vision 연구로서 새 방법이나 넓게 재사용 가능한
새 통찰이 약하면 MICCAI, MIDL, TMLR, PoPETs 등의 독자층과 더 자연스럽게 맞을 수 있다.
반대로 CVPR main을 우선 목표로 한다면 생성모델 자체의 환자 단위 프라이버시 문제를
해결하는 기여를 전면에 둔다.

## 6. 권장 CVPR형 논문

### 6.1 권장 한 문장

> We introduce a patient-aware differentially private latent diffusion method that aggregates
> multi-image contributions before clipping and noising, together with a unit-audited release
> protocol that decides when image-DP checkpoints can be reused and when patient-DP retraining
> is required.

한국어로는 다음과 같다.

> 환자의 복수 영상 기여를 clipping과 noise 주입 전에 하나의 기여로 집계하는
> patient-aware DP latent diffusion 방법을 제안하고, 기존 image-DP checkpoint를 재사용해도
> 되는 경우와 patient-DP로 재학습해야 하는 경우를 판정하는 unit-audited release protocol을
> 함께 제시한다.

### 6.2 가장 강한 신규 기여 구성

1. **Diffusion-specific patient-DP method**
   - patient-first sampling 또는 patient contribution aggregation
   - 환자별 복수 image/timestep contribution을 집계한 뒤 clipping/noising
   - patient-level `(epsilon, delta)`의 명시적 보장
   - 기존 P3SGD/user-wise DP의 단순 재명명을 피하고 diffusion 구조에서 필요한 새 설계와
     효용 개선을 보인다.
2. **Tight accounting 및 reuse frontier**
   - image-DP의 generic group bound
   - 가능한 경우 DP-SGD-specific tighter conversion
   - 동일 target patient privacy에서 converted image-DP와 native patient-DP의 공정 비교
   - `reuse allowed`, `claim downgrade`, `retraining required`의 사전 규칙
3. **Patient-level empirical privacy**
   - patient membership inference
   - exact/near-duplicate 및 memorization audit
   - 환자당 영상 수 `K=2/5/10`에 따른 위험 변화
   - 공격이 null이어도 formal claim을 대신하지 않는다는 해석
4. **Vision 및 clinical utility**
   - 생성 품질과 다양성
   - downstream 분류/증강 효용
   - subgroup·lesion coverage와 전문가 또는 임상적으로 타당한 proxy
   - privacy–utility–compute frontier
5. **일반성**
   - 가능하면 두 개 이상의 의료영상 데이터셋 또는 서로 다른 patient multiplicity 조건
   - 최소한 하나의 외부 데이터 또는 cross-dataset 검증
   - 공개 ISIC 실험을 실제 비공개 임상 배포 증거로 과장하지 않음

새 알고리즘이 없더라도 매우 넓고 엄밀한 benchmark와 개념적 발견으로 CVPR에 도전할 수는
있다. 그러나 현재 상황에서는 **새 patient-aware diffusion mechanism + unit-aware audit**의
결합이 가장 안전하고 강한 경로다.

## 7. AAAI에서 재사용할 것과 새로 만들어야 할 것

### 재사용 가능한 기반

- privacy unit·sampling unit·clipping unit·accounting unit의 구분
- evidence schema와 certificate/receipt 형식
- verifier, conformance test, artifact lineage
- 과장된 privacy claim을 탐지하는 decision logic
- reproducible packaging과 독립 검증 구조

### 의료영상 논문의 독립 신규성으로 남겨야 할 것

- patient-aware latent diffusion training mechanism
- image-DP에서 patient-DP로의 tight conversion 및 실패 조건
- 환자 단위 공격과 생성 memorization 평가
- 의료영상 utility 및 임상적 proxy
- checkpoint reuse와 native retraining의 의사결정 경계
- 복수 영상 환자 구조가 만드는 새로운 empirical phenomenon

## 8. 현재 로컬 작업과의 연결

이미 준비된 다음 자산은 이 방향의 실행 가능성을 높인다.

- patient-first split과 `K=2/5/10` cap
- `B0/M0/M1/M2` 비교 틀
- exact SD 2.1 snapshot 및 LoRA smoke gate
- ISIC loader와 latent encoding 확인
- audit/certificate 계열 코드와 artifact discipline
- PP-Mark compatibility 진단

그러나 다음은 아직 완료된 결과로 취급하지 않는다.

- 실제 patient-DP generator 학습
- tight group accounting의 정식 증명·검증
- patient-level attack와 memorization 결과
- confirmatory multi-seed utility 결과
- PP-Mark가 결합된 model-bound release receipt

RTX 3070 검증은 skeleton/pilot 실행 가능성의 근거일 뿐이다. CVPR용 confirmatory 학습은 더
큰 GPU 또는 계산 자원이 필요할 수 있다.

## 9. 최종 판단

**주제 선정은 CVPR에 적당하다.** 특히 환자 한 명이 여러 영상을 갖는 의료 데이터에서는
image-level DP와 patient-level 보호 사이의 차이가 실제 배포 주장을 바꾸므로 문제 자체가
인위적이지 않다. 공개 가능한 synthetic medical image와 memorization 문제도 computer vision
독자에게 직접 보이는 결과를 만든다.

다만 제목과 초록이 `AAAI audit의 의료영상 확장`으로 읽히면 약하다. CVPR형 독립 논문은
다음 구조로 고정하는 것이 좋다.

> **새 patient-aware private diffusion method를 제안하고, AAAI에서 만든 audit 체계로 그
> 방법과 기존 image-DP checkpoint의 patient-level 배포 가능 경계를 검증한다.**

따라서 AAAI의 채택 여부는 이 연구의 진행 가능성을 결정하지 않는다. 채택되면 명확한
후속·확장 관계로 연결하고, 탈락하면 feedback을 흡수하되 별도 vision method paper로
완결한다.

## 10. 상태 경계

- 이 문서는 방향과 투고전략 기록이다.
- frozen experiment contract를 변경하지 않는다.
- generator/DP training 시작을 승인하지 않는다.
- 활성 학위논문 LaTeX·초록을 수정하지 않는다.
- CVPR 2026 또는 이후의 채택 가능성을 보장하지 않는다.
