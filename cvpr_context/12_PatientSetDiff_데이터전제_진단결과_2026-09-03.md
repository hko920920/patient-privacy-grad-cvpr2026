# Patient-Set Private Diffusion 데이터 전제 진단

- 기록일: 2026-09-03
- protocol SHA-256: `02B291BA61F90E1B0E5011AAC6A687BCC6448B949520C9D12C2ECE49A440BBA4`
- 형식 판정: **`FAIL_PATIENT_SET_DATA_PREMISE` (5개 중 4개 통과)**
- 해석: **절대 cosine-gap gate는 실패했으나 scale-free patient signal과 record variation은 강함**

## 질문

joint synthetic patient set을 만들려면 반복 X-ray에 (a) 같은 환자의 안정된 시각적 구조와
(b) records 사이의 실제 소견 변화가 동시에 있어야 한다. 둘 중 하나가 없으면 patient-set
generation은 단순 복제 또는 무관한 이미지 묶음이 된다.

## outcome-blind 표본과 probe

- NIH K10 public development
- 모든 이전 gradient 실험 160 patients 제외
- 4장 이상인 target/control 각 40명, 총 80 patients·320 images
- 공개 DINOv2 ViT-B/14, frozen weights
- full-field 518 x 518, feature vector 저장 없음

## 결과

| 결과 | 값 | 사전 조건 | 판정 |
|---|---:|---:|---|
| within minus cross-patient mean cosine | `0.011060` | `>=0.05` | FAIL |
| same-patient balanced similarity AUC | `0.877852` | `>=0.75` | PASS |
| retrieval R@1 | `0.65625` | chance의 5배 이상 | PASS (`69.78배`) |
| 여러 exact label set을 가진 환자 | `0.7125` | `>=0.30` | PASS |
| within-patient label Jaccard distance | `0.500799` | `>=0.10` | PASS |

추가로 same patient이면서 exact finding label이 다른 pair의 평균 cosine은 `0.986063`이고,
different patient이면서 target과 exact finding label이 같은 pair는 `0.974805`였다. 따라서 DINO
probe가 병명만 보고 같은 환자를 찾았다는 설명만으로는 결과가 설명되지 않는다.

## 판정과 다음 경계

사전 문서는 다섯 조건을 모두 요구했으므로 status를 PASS로 바꾸지 않는다. 절대 cosine 차이
`0.05`는 전체 cosine이 0.97 이상에 몰린 이 probe에서 scale-dependent하고 과도한 기준이었다.
그 사실을 결과 뒤에 이용해 같은 cohort의 threshold를 낮추지 않는다.

다만 AUC, retrieval, label-change, label-Jaccard 네 개의 scale-free 결과와 MedReID 선행을 보면
patient-set 연구문제의 실제 데이터 전제는 유력하다. 다음 검정은 이번 80명과 겹치지 않는
2--3 record patient cohort에서 AUC·retrieval 중심 기준을 새로 고정해야 한다. 그것이 통과하기
전에는 private training으로 진행하지 않는다.

## 산출물

- protocol/code: `code_working/patient_set_diffusion/`
- report: `code_working/_reports/nih_cxr14_patient_set_premise_v1_001/report.json`
- selection SHA-256: `87B356A6175493E309DE2D84E8D0769B4764005BEE809AA8EB5CF5D5EF544AB0`

## 후속 독립 확인(같은 날)

이 문서의 v1 status는 그대로 `FAIL`이다. 다만 v1의 80명과 앞선 160명을 전부 제외하고,
정확히 2--3 records인 새 160명·400 images에서 결과 전 scale-free 기준을 동결한 v2는 5/5
`PASS_PATIENT_SET_DATA_PREMISE_CONFIRMATION`이었다. pooled AUC `0.880605`, 네 cell AUC
`0.863333--0.908472`, patient ordering wins 전체·각 cell `0.975`, global R@1 `0.48`
(`119.7x` chance), record variation fraction `0.75625`, label Jaccard `0.582240`이다.

이 독립 결과는 v1 threshold를 사후 수정한 것이 아니며, architecture smoke로 진행할 근거만
제공한다. 상세 protocol, artifact hash와 뒤이은 Q=2 SetAdapter PASS는
`13_PSPD_독립확인_SetAdapter_구조smoke_2026-09-03.md`를 따른다.
