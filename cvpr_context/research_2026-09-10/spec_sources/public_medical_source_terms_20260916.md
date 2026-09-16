# 공개 의료 기반모델 자료 후보: 출처·접근 범위 확인

2026-09-16 공식 제공 경로를 확인했다. 이번은 다운로드나 학습 실행이 아니라 자료 경계 선택이다.

## 이번 pilot의 우선 선택

현재 로컬 NIH PA 이미지 중 다른 연구 역할과 환자가 겹치지 않는 집합으로 별도 공개 학습/선택/확인 분할을 만든다. 같은 NIH 원천 안의 독립 역할 분할이며 **외부 병원 데이터나 별도 기관 일반화 설정이라고 부르지 않는다**. 공개32명은 작은 head의 공개 기준선용으로 남긴다.

Google Cloud의 NIH 데이터 제공 문서는 NIH 영상 이용 제한이 없다고 안내하면서, NIH 다운로드 링크·CVPR2017 원논문 인용·NIH Clinical Center 출처 표기를 요구한다. [공식 데이터 제공 문서](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest). 원 NIH Box 링크는 이번 웹 도구에서 열리지 않았으므로 원 README를 새로 읽었다고 하지 않는다. 실제 로컬 취득 출처와 hash는 별도 명부 감사에 따른다.

같은 문서의 Cloud Storage 경로는 requester-pays로 안내된다. 이번 로컬 재사용 계획에는 신규 유료 다운로드나 클라우드 이용이 없다.

## 비교한 외부 선택지

- CheXpert는 Stanford가 제공하는 별도 의료기관 자료다. 공식 페이지에는224,316영상/65,240환자와 다운로드 경로가 있지만, 이번에 읽힌 Terms/License 본문은 비어 있었다. 이를 보고 사용 조건을 모두 확인한 즉시 사용 후보라고 처리하지 않는다. 현재 pilot에 이 자료를 새로 받는 의존성을 추가하지 않는다. [Stanford AIMI 제공 페이지](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays).
- MIMIC-CXR-JPG는 명칭상 publicly available이더라도 credentialed access이고 DUA가 필요하다. 공식 페이지는 재배포 제한 등의 조건을 설명한다. 따라서 로컬에 MIMIC 폴더가 있다는 사실만으로 무제한 공개 보조자료로 분류하지 않는다. 이번 분할/학습에는 사용하지 않는다. [PhysioNet 공식 v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).

## 기반 checkpoint와 DP 해석

이번 후보는 기존 캐시에 있는 `Manojb/stable-diffusion-2-1-base`의 고정 snapshot을 초기화로 사용한다. 해당 배포 페이지는 CreativeML Open RAIL++-M을 표시한다. 현재 upstream StabilityAI 페이지는 웹 요청에서401을 반환했으므로 원 upstream weights와 새로 bitwise 대조했다고 하지 않는다. 기존 로컬 snapshot/source hash를 재현 기준으로 유지한다. [실제 사용 checkpoint 배포 카드](https://huggingface.co/Manojb/stable-diffusion-2-1-base).

새 공개 domain adaptation에 protected-role 환자가 들어가지 않았다는 것은 명부로 확인할 수 있다. 기초 모델의 전체 웹 사전학습 자료에 그 환자가 전혀 없었다는 보장은 별개이며 이번에 확보하지 않았다. 향후 patient-DP 주장은 고정된 기반모델·공개 보조자료 아래 새 private adaptation의 인접성에 관한 것으로 범위를 명시해야 한다. 기존 비DP M1의 사적 노출을 작은 DP head로 제거했다고 주장할 수 없다.

기관이 다른 데이터가 있어야만 pilot이 가능한 것은 아니다. 먼저 통제된 NIH 역할 분할에서 구성의 효용을 확인하고, 외부 자료/분포 차이는 그 결과와 최종 주장에 맞춰 별도로 검토한다.
