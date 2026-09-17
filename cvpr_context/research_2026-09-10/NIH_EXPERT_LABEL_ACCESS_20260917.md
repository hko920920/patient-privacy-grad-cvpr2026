# NIH 전문가 라벨 접근 결과와 공개 부록의 실제 표본 수

2026-09-17. 큰 단계 2 / 방향 1. **즉시 자료 확보에는 부정적이다. 공식 파일은 접근하지 못했고 환자 중복 감사도 미완료다.** 다만 공개 부록을 실제 확보해, 이 자료의 크기에 대한 중요한 판단 근거를 얻었다. 새 모델 추론·생성·학습·DP는 없다.

## 접근을 실제로 시도한 결과

1. [공식 NIH 안내](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels)가 연결한 요청 양식은 두 범주형 질문뿐이었다. 기관 유형 `Academia`, 목적 `Benchmarking different ML methods`로 익명 다운로드 절차를 완료했다. 이름·이메일·연락처·로그인 계정은 제출하지 않았다.
2. 응답은 [공식 라벨 버킷](https://console.cloud.google.com/storage/browser/gcs-public-data--healthcare-nih-chest-xray-labels)을 안내했다. 이것이 실제 읽기 권한을 부여한다는 뜻은 아니었다.
3. 익명 버킷 목록과 문서에 명시된 두 CSV의 object GET이 모두 HTTP 403을 반환했다. 파일을 다운로드하지 못했다. 에러는 `storage.objects.get` 접근 거부이며, 객체 부재도 같은 응답일 수 있다고 표시된다. 따라서 버킷 삭제나 파일 부재를 확정하지 않는다.
4. 사용자는 버킷에서 Access denied를 보고했고, 이어 로그인용 개별 파일 URL도 403이라고 보고했다. 사용자 계정의 로그인 상태·IAM 정책 자체를 원격으로 조사한 것은 아니다.
5. [Google 공식 브라우저 다운로드 방식](https://docs.cloud.google.com/storage/docs/request-endpoints#authenticated_browser_downloads)에 따라 `storage.cloud.google.com` 직접 링크도 확인했다. 익명 요청은 로그인으로 리다이렉트된다. 이 경로는 권한 확인을 받는 정상 다운로드이며 권한을 우회하는 주소가 아니다. **로그인용 직접 링크까지 거부된 현재에는 배포자에게 유효한 링크나 읽기 권한을 문의하는 것이 다음 접근 절차다.** [공식 권한 설명](https://docs.cloud.google.com/storage/docs/collaboration)

직접 파일 경로:

- [test_labels.csv](https://storage.cloud.google.com/gcs-public-data--healthcare-nih-chest-xray-labels/all_findings_expert_labels/test_labels.csv)
- [test_individual_readers.csv](https://storage.cloud.google.com/gcs-public-data--healthcare-nih-chest-xray-labels/all_findings_expert_labels/test_individual_readers.csv)

`Documents`, `Downloads`, `Desktop`의 관련 파일명 검색에서 이 라벨 CSV는 찾지 못했다. `Documents/FeedbackHub` 일부 폴더는 파일 목록 접근이 거부되었으므로 모든 로컬 저장소에 없다는 주장은 하지 않는다.

## 라벨 CSV 없이도 확인한 중요한 사실

[원논문 공개 부록](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-021-93967-2/MediaObjects/41598_2021_93967_MOESM1_ESM.docx)을 실제 내려받았다. **Supplementary Table 4**가 보고한 CXR-14 전문가 기준 양성 영상 수는 다음과 같다.

| 소견 | 양성 영상 수 | 현재 판단 |
|---|---:|---|
| Emphysema | 7 | 환자 중복·기존 역할 제외 전에도 양성 환자는 최대 7명. 이 자료 하나로 기존 예시인 질환당 20/32/64명을 확보할 수 없음 |
| Pneumothorax | 136 | 기흉 평가자료 후보 규모는 있으나 환자 수·중복·사용 역할은 미확인 |
| Effusion | 226 | control 후보 규모는 있으나 두 target과의 동반 여부 및 환자 중복은 미확인 |

이것은 **810장의 전체 전문가 라벨 중 논문에 보고된 영상 수**다. 우리가 원시 라벨을 다시 세어 얻은 값이나 독립 환자 수가 아니다. XML 추출과 python-docx 표 읽기로 같은 수를 확인했다. 처음 python-docx 교차 확인은 표 제목의 선행 줄바꿈 때문에 표를 찾지 못했다. 앞뒤 공백을 제거하여 표를 식별했으며 수치나 기준은 변경하지 않았다.

부록의 DLS AUC는 해당 연구의 정상/비정상 시스템 평가다. 표에 높은 수치가 있다는 이유로 새 질환별 생성 평가기가 확보됐다고 해석하지 않는다. 공개 표만으로 E/P 동시 양성 환자 수, 기존 weak label 검증80명의 오류 원인, train/test 중복도 계산할 수 없다.

## 완료한 것과 아직 하지 못한 것

| 항목 | 상태 |
|---|---|
| 공식 접근 안내·양식·버킷·직접 객체 경로 | 실제 확인, 파일 접근 실패 |
| 공개 부록 target 양성 수 | 확인: 폐기종 7장 / 기흉 136장 |
| expert label CSV 및 개별 전문의 판독 CSV | 미확보 |
| expert label 환자별 수·E/P 동시 양성 | 미계산 |
| backbone640 / public32 / private80 / dev40 중복 | 미계산 |
| evaluator 소비160 / reserved4213 / 잠긴 졸논 자료 중복 | 미계산 |
| 새 전문가 라벨을 개발·최종 평가에 배정 | 하지 않음 |
| 생성·분류기·DP 실행 | 모두 0 |

대조할 기존 명부의 존재와 해시는 확인했다. 공개 backbone 계획 명부 전체905명에는 train640, selection128, confirmation128, reserve9가 들어 있다. 905명을 모두 학습 환자로 세면 안 된다. 실제 head 명부는 public32/train80/eval40, 두 evaluator 새 소비 집합은 합집합160명, 별도 CVPR reserved confirmation은4213명이다. **이 숫자는 기존 명부 크기이며 expert 810장과의 교집합 결과가 아니다.**

## 연구 판단

**이 특정 파일은 유일한 필수 자료가 아니다. 신뢰할 수 있고 환자 분리가 확인된 실영상 평가 기준이 필요한 것이다.** 같은 NIH 출처의 연결 가능한 전문가 주석이라는 장점 때문에 확보를 시도할 가치는 있으나, 접근이 풀려도 폐기종 표본 문제가 자동으로 해결되지 않는다.

따라서 “810장 전문가 주석을 받으면 두 질환 공동 downstream 검증을 바로 실행한다”는 계획은 채택하지 않는다. 기흉 중심의 별도 연구 명세, 추가 폐기종 판독/자료 또는 다른 신뢰할 효용 정의를 비교할 근거가 생긴 상태다. 어느 하나를 사후에 남겨 기존 두 평가기의 공동 가설 실패를 성공으로 바꾸지 않는다.

사용자 제안의 downstream 추가 효용은 여전히 타당한 후보다. 그러나 여덟 학습 arm을 바로 실행할 단계는 아니다. 먼저 사용 가능한 real-test와 주장 범위를 정하고, 비DP private/pooled synthetic 대 public synthetic의 핵심 비교를 설계해야 한다. Private real non-DP 추가는 진단용 비교군이며 보장된 성능 상한이 아니다.

접근 요청의 시간은 배포자 회신에 달려 있어 예상할 수 없다. CSV가 확보된 후 metadata 환자 중복 감사는 20–30분 예상이다. 지금은 이 자료를 독립 real-test로 채택하지 않고 이전192장·두 classifier 실패·reserved confirmation·졸논 역할을 보존한다.

## 공식 문의처와 준비한 문안

[원논문](https://www.nature.com/articles/s41598-021-93967-2)에 공개된 교신저자는 Daniel Tse (`tsed@google.com`), Po-Hsuan Cameron Chen (`cameronchen@google.com`), Shravya Shetty (`sshetty@google.com`)다. 이는 출판물의 연락처이며 현재 메일 수신 가능 여부까지 확인한 것은 아니다. 우선 한 교신저자에게 현재 배포 경로를 문의하는 문안을 준비했다.

[메일 문안](NIH_EXPERT_LABEL_ACCESS_REQUEST_20260917.txt). **아직 발송하지 않았다.** 이 환경에는 연결된 메일 발송 도구가 없으며, 사용자를 대신해 외부인에게 보낼 때는 명시적인 발송 지시가 필요하다. 현재는 접근 방법 문의에 대한 구체적인 초안을 제공한 상태다.

로컬 접근 기록·부록·입력명부 hash·상태: `code_working/_reports/nih_expert_label_inventory_20260917_v1`. 이 보고서는 신규 모델 실험 결과가 아니다. 마지막 실제 모델 결과는 [CheXzero 검증](TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md)으로 유지한다.
