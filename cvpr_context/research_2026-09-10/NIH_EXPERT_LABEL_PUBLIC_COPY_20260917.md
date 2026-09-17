# NIH 전문가 라벨 공개 가공본 확보와 접근 결론 정정

2026-09-17. 큰 단계2 / 방향1의 자료 접근 작업. **자료 확보에는 좋은 결과다. 직접 연락 없이 810장·532명의 14소견 라벨을 담은 공개 가공본을 실제 다운로드하고 CSV로 추출했다.** 이전의 ‘직접 연락이 다음 유일한 경로’라는 결론은 공개 재배포본을 충분히 찾기 전에 내린 성급한 판단이었다. 공식 서버의403은 유지되지만 자료 접근 전체가 막힌 것은 아니다.

## 실제 확보한 자료

- [공개 가공본의 고정 커밋 파일](https://github.com/manwithacat/Radiology-for-Dummies/blob/b49f9519dba7237529cf6975ee9f268e4c370de8/data/processed/metadata_with_optimal_labels.parquet)
- [로그인 없는 직접 다운로드](https://raw.githubusercontent.com/manwithacat/Radiology-for-Dummies/b49f9519dba7237529cf6975ee9f268e4c370de8/data/processed/metadata_with_optimal_labels.parquet)
- [가공 코드](https://github.com/manwithacat/Radiology-for-Dummies/blob/b49f9519dba7237529cf6975ee9f268e4c370de8/jupyter_notebooks/02_exploratory_data_analysis.ipynb)

전체 파일은112,104행으로 원 NIH 라벨·Nabulsi 전문가 라벨·다른 라벨 및 통합 열이 함께 있다. **통합 `optimal_*` 열을 전문가 정답으로 사용하지 않았다.** `nabulsi_*` 원출처 열14개가 채워진810행만 분리했다. 이 별도 열은 저장소 코드상 공식 `all_findings_expert_labels/test_labels.csv`의14소견을 영상ID로 결합하고 `YES/NO`를 `1/0`으로 변환한 것이다. 가공 코드는 읽었으며 실행하지 않았다.

## 확인한 일치와 아직 확인하지 못한 것

| 검사 | 결과 |
|---|---|
| 실제 HTTP 다운로드 | 성공, 익명 요청200 |
| 커밋 고정 URL의 파일과 최초 다운로드 내용 | 동일 |
| 전문가 출처 행 |810장, 영상ID 중복0 |
| 환자ID |532명, 영상명 환자번호와 일치 |
| 영상view |전부PA |
|14소견 값 |모든810행에서 결측 없이0/1 |
| 공식 부록 Table4의14소견별 양성 수 |14개 모두 일치 |
| 별도 후속 논문 저자 저장소의 영상ID 집합 |810개 전부 일치 |
| 공식 원본 CSV와 바이트 또는 모든 행의 직접 대조 |미수행: 원본 접근은 여전히403 |
|14소견의 전문의별 원본 개별 판독 |미확보 |
|기존 연구 역할과 환자 교집합 |이번 접근 작업에서는 미계산 |

양성 영상 수에는 폐기종7장·기흉136장·흉수226장이 포함된다. 이는 이제 공개 가공본에서도 실제 집계한 수이며 [공식 부록](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-021-93967-2/MediaObjects/41598_2021_93967_MOESM1_ESM.docx)과 일치한다. 집계와ID 일치는 강한 정합성 근거지만 모든 개별 라벨이 공식 원본과 같다는 절대적인 증명은 아니다.

교차 확인한 별도 자료는 [ICML2025 후속 논문](https://arxiv.org/abs/2506.05636)의 [저자 저장소](https://github.com/markellekelly/consensus/tree/38f2ceb1040b21a0e956660594eec3f438a90e34/data/nih)다. 그 저장소의 전처리는 원래 개별 판독 CSV에서 `Abnormal`만 가져오며, 데이터에는 영상ID와 다섯 전문의의 정상/비정상 판독이 남아 있다. **이 자료는14소견 라벨 자체의 별도 대조본이 아니다.** 또한 저자 코드의5명 다수결을 공식 지정3명 다수결 정답과 같다고 간주하지 않는다.

공개 가공본에는 다른 출처의 broad finding을 여러 질환으로 펼친 통합 열도 있으므로 전체를 임상 정답으로 채택하면 안 된다. 이번 추출은 해당 열들을 전부 제외했다.

## 함께 확인한 다른 경로

- 공식XML 직접 객체·가상호스트는403, JSON API·download API는401. 단순 주소 형식 변경으로 접근이 해결되지 않았다.
- Google Health 및 imaging-research 공개 저장소에서는 해당 원본 라벨 파일을 찾지 못했다.
- 별도 후속 논문 저자 저장소는 정상/비정상 파생본을 제공한다.
- 확인한 원본CSV 경로의Git 이력조회에는 원본 파일 커밋이 없었다. 이는 조사한 경로의 결과이며 모든 공개 저장소에 원본이 없다는 증명이 아니다.
- Internet Archive 조회는시간초과였다. 보관본이 없다는 결론을 내리지 않는다.

## 현재 결정

**메일 회신을 기다려야만 자료 검토를 진행할 수 있는 상태는 아니다.** 추출한 후보 명부로 환자 중복과 평가자료 규모를 검토할 수 있다. 공식 원본·전문의별14소견 판독 확보는 추가적인 출처 확증 수단으로 남는다.

다만 확보 성공과 평가자료 충분성은 다르다. 폐기종 양성7장이라는 제한은 그대로이고, 기존 모델·잠긴 졸논 자료와의 중복도 아직 확인해야 한다. 이 자료를 곧바로 최종 real test로 채택하거나 실패한 두 evaluator 실험을 재개하지 않는다. 다음 필요한 작업은 이 후보의 환자 역할 중복과 사용 가능한 주장 범위를 검토하는 metadata 감사이며 예상20–30분이다.

로컬 산출물은 `code_working/_reports/nih_expert_label_public_routes_20260917_v1/nih_all14_from_public_mirror_810.csv`와 같은 폴더의 `recovery_record.json`, `extract_public_mirror.py`다. 원파일·소스코드·커밋·SHA와 추출 범위를 기록했다. 새 환자영상 열람·모델추론·학습·생성·DP는0이고 메일도 보내지 않았다. 과거403접근 기록과 모든 모델 실패 결과는 보존한다.
