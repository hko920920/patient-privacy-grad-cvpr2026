# 고한경 박사학위논문 작업본

> 세션을 이어서 작업할 때는 먼저 `WORKLOG.md`의 현재 단계와 남은 작업을 확인합니다.

## 현재 권위와 정지점 — 2026-09-09

현재 상태와 과거 기록의 supersession은 먼저 `CURRENT_STATUS.md`를 확인합니다. NIH X-ray 원본
42,423장과 B0 448장 확보·검증까지 완료됐고, B0는 off-domain comparator로 보존합니다. full K5
matrix와 M0는 시작하지 않았습니다. 학위논문 통합 실험의 다음 mutating 단계는 별도 승인 후
**M0만 먼저** 실행·생성·평가하는 것입니다.

B0보다 앞서 K5 `private_train` 역할 partition에서 세 DP arm을 각각 4 steps 실행한 별도
`RESEARCH_ONLY` dry-run이 있었습니다. 따라서 B0 시간순서는 모든 private-role optimizer update가
아니라 **full 4,000-step K5 matrix 이전**으로 제한합니다. 상세 정정은 `CURRENT_STATUS.md` Section 3을
따릅니다.

`CVPR 주제 탐색/`은 별도 논문기획 스트림입니다. 최신 authority는 18번 SOTA 대조 문서이며,
학위논문 동결 실험계약이나 M0 실행을 자동으로 변경하지 않습니다.

작업 제목: **Claim-Aligned Assurance for Trustworthy AI: Auditable Privacy and Secure Public Verification**

이 폴더는 서강대학교 가상융합전문대학원 가상융합테크놀로지전공
박사학위논문 집필용 작업 공간입니다.

## 기준

- 편집 규격: `Technology_v1.14/Template_Doctor_Dissertation`
- 구조ㆍ분량 참고: `김태훈 교수님_박사졸논 자료/학위논문_최종.pdf`
- 편집 규격과 참고 논문이 충돌하면 최신 Technology 템플릿을 우선합니다.

## 파일 구성

- `thesis.tex`: 전체 문서와 제출 정보
- `CURRENT_STATUS.md`: 최신 상태, 두 연구 스트림, supersession과 2026-09-09 기록 감사 정정
- `WORKLOG.md`: 세션 인수인계·참고 근거·수정 내역·남은 작업 통합 기록
- `notation.tex`: 장 간 기호 충돌을 방지하는 학위논문 공통 표기
- `notes/terminology_notation_claim_policy.md`: 용어·보증 경계·주장 강도 기준
- `notes/revision_posthoc_only_2026-08-25.md`: 사전 연구 제외 이후 전면 개편 근거·수치·QA 기록
- `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md`: 비공개 이미지 생성 모델 수명주기로 통합하는 신규 방향, 현 본문과의 간극, 기승전결 전면 재설계, 연구·집필 마일스톤
- `notes/chapter01_content_map.md`: Chapter 1 선행연구·통합 주장·분량·QA 근거표
- `notes/chapter02_content_map.md`: Chapter 2 수식·그림·표·실험 결과 배치표
- `notes/chapter03_content_map.md`: 제외된 사전 DP 장의 보존용 과거 배치표
- `notes/chapter04_content_map.md`: PP-Mark 장의 과거 번호 기준 근거표
- `notes/chapter05_content_map.md`: 통합 결론의 과거 번호 기준 근거표
- `notes/abstracts_content_map.md`: 국·영문 초록 수치 대응·키워드·분량·QA 근거표
- `sgmeta_report.cls`: 작업본 전용 클래스 파일
- `frontmatter/`: 감사의 글과 국ㆍ영문 초록
- `chapters/`: 장별 본문
- `figures/`: 그림 원본
- `tables/`: 대형 표 조각 파일
- `refs.bib`: 참고문헌 데이터
- `build/`: 조판 결과와 보조 파일
- `template_reference/`: 수정하지 않는 원본 템플릿 사본

## 현재 장 구성

1. Introduction
2. Post-Execution Auditing of Privacy-Unit Claims
3. Secure Public Verification of Generative-AI Provenance: PP-Mark
4. Conclusion

이더리움 이종 그래프, 메타버스 공간 인증, LLM 기반 데이터 증강 연구는
Chapter 1의 연구 궤적에만 배치합니다. Chapter 2는 완료된 window-based DP-SGD
실행에서 요청된 privacy-unit 문구를 증거에 따라 허용·변환·차단하는 사후 감사를,
Chapter 3은 PP-Mark의 통계적 탐지와 proof-bound 공개 출처 검증을 다룹니다.
Chapter 4는 두 연구축을 Claim-Aligned Assurance 원리로 통합합니다.

기존의 pre-execution route-registration 장은 현재 학위논문에서 제외했습니다.
원본 `chapters/03_evidence_sealed_registration.tex`은 연구 이력 보존을 위해 남겨
두었지만 `thesis.tex`에서 입력하지 않으므로 목차·본문·참고문헌·최종 PDF에는
포함되지 않습니다.

2026-08-25 기준으로 Chapters 1--4, 국·영문 초록, 참고문헌, 그림·표 목록과
전체 조판 QA를 완료했습니다. Chapter 2는 main paper와 Supplement를 함께 반영하여
본문 13--43쪽의 31 B5 pages, Chapter 3 PP-Mark는 44--76쪽의 33 B5 pages입니다.
최신 `build/thesis.pdf`는 104 physical pages, B5 JIS, 5,001,029 bytes입니다.
LaTeX 오류, 미정의 인용·참조, overfull box는 없습니다. 미완성 추가 실험은
Chapter 4의 `Future Experimental Validation`에서 명시적인 후속 검증 설계로만
구분하며 완료 결과로 기술하지 않습니다.

2026-08-31에는 다수 사용자의 비공개 이미지로 학습·미세조정한 생성 모델을 하나의
대상으로 두고, privacy-unit audit 결과를 정확한 모델 checkpoint에 결속한 뒤
PP-Mark가 생성 이미지와 함께 공개 검증하게 하는 연구 확장안을 기록했습니다. 이
방향은 현재 두 축의 공통 원리 통합보다 강한 end-to-end artifact integration을
요구하므로, 상세 간극과 단계별 gate는 위 신규 note를 따릅니다. 이번 기록 단계에서는
활성 LaTeX 본문과 104쪽 PDF를 변경하지 않았습니다.

2026-09-02에는 이 출발 상황이 실제 연구 환경인지 먼저 재검증했습니다. private client
image collections로 shared diffusion model을 학습하는 federated 연구, sensitive
images와 cardiac MRI에 대한 DP diffusion 학습, diffusion model의 training-image
extraction, image-level과 실제 patient-level 보장의 차이가 모두 기존 문헌에 존재함을
확인했습니다. DP는 raw images를 이미 수집한 training operator에게 제출자를 숨기는
방법이 아니라 released model·API·outputs의 관찰자에 대한 membership와 influence
보장입니다. operator에게도 raw data를 숨겨야 하면 federated learning과 secure
aggregation 등을 별도로 요구합니다. 상세 근거와 허용·금지 문구는 위 신규 note
Section 1.1과 `notes/terminology_notation_claim_policy.md` Section 12를 따릅니다.

새 방향을 채택할 때는 현재 Introduction 뒤에 통합 장만 덧붙이지 않습니다. 검증된
기술 결과는 보존하되, 중심 시나리오·문제 설정·RQ·contribution·장 전환·Conclusion과
초록은 하나의 private-image generative lifecycle이 처음부터 이어지도록 다시
설계합니다. 2026-09-02 당시 첫 설계 단계는 LaTeX 부분 수정이 아니라 narrative
architecture와 Chapter 1 문단 outline의 확정이었고, 이후 데이터·모델·실행 gate와 B0까지
진행했습니다. 현재 정지점은 이 문서 상단과 `CURRENT_STATUS.md`를 따릅니다.

현행 104쪽 기준본만을 제출할 경우 남은 작업은 영문 저자명, 지도교수·심사위원·날짜
등 제출 메타데이터와 선택 사항인 감사의 글을 실제 정보로 확정하는 것입니다. 신규
통합 실증 방향에서는 사전 데이터·모델·실행 gate와 B0가 완료됐지만 M0와 full DP arms,
공격, receipt, PP-Mark 통합은 아직 없습니다. 그 결과를 동결한 뒤에만 Introduction,
신규 통합 장, Conclusion과 초록을 순서대로 개편합니다.

## 조판 순서

참고문헌을 포함한 전체 조판 순서는 다음과 같습니다.

1. `pdflatex -output-directory=build thesis.tex`
2. `bibtex build/thesis`
3. `pdflatex -output-directory=build thesis.tex`
4. `pdflatex -output-directory=build thesis.tex`

Windows에서는 프로젝트 루트에서 `./build.ps1`을 실행하면 위 순서를 자동으로
수행합니다.

## 우선 입력할 정보

- 국문ㆍ영문 논문 제목
- 영문 저자명
- 지도교수명
- 제출일, 심사일, 졸업월
- 심사위원 5명
- 국문ㆍ영문 키워드
- 학번 표기 여부

제출자·지도교수·심사위원·날짜·초록·핵심어 자리표시자는 실제 정보를 받은
뒤에만 확정합니다.
