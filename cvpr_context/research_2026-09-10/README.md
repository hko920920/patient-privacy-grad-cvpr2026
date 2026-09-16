# 환자 DP diffusion 선행연구 직접 분석

[U 실험 실행 현황](u_execution_status.html): 사용자 승인 후 실제 사진 선정·독립 분할 검증·입력 준비·새 코드와 파일럿 학습의 결과를 갱신한다. 아래 계획 문서의 작성 당시 미실행 표기보다 현재 실행 상태는 이 기록을 따른다. [Markdown](U_EXECUTION_STATUS.md).

[기존 U 8명 판별 결과 분석](u_eight_patient_analysis.html): 후보 AUC 0.6875/0.3125, 환자 참여 교환 후 순위 동일. 현재 표본에서는 일관된 후보 효과를 확인하지 못했다. 기존 값만 분석했으며 추가 GPU 실행·환자 확대는 없다.

[U 파일럿 정밀 검토·해석 정정](u_pilot_review.html): 구현·자료·학습·통계를 대조했다. 보정항 실패라는 원인 추정을 철회하고, 빠진 비영점 미분/최적화 검증과 E/U 기본 대조, 미실시 fitting 단계를 구분했다. 기존 입력의 수치 진단 사양을 준비했으며 GPU 실행은 하지 않았다.

[U 중심 실행 계획](u_execution_plan.html): 2026-09-14. 학습 2장·별도 관측 2장의 첫 파일럿안, 환자 참여를 교환한 두 모델 진단, 기존 방법과 새 후보의 비교, 진행·수정 기준 및 DP 보호 비교 순서. 다음 작업은 실제 파일 기준의 적격 수와 분할표 확정이다. [Markdown](U_EXECUTION_PLAN.md). 아직 분할·구현·학습 실행 결과는 아니다.

[U 조건을 중심으로 한 비교군 검토](baseline_scope_review.html): 2026-09-14 대화의 핵심인 관측 영상과 학습 영상의 간극을 유지하며, 네 가지 문헌 역할·FACE-AUDITOR의 미사용 동일인 영상 선행·MoFit/Quantile 등의 정확한 비교 역할·환자 오탐 보정과 비용을 검토했다. 실험 실행·성능 확인은 아니다. [Markdown](BASELINE_SCOPE_REVIEW.md).

[제한 접근 감사자 설정 검토](auditor_framing_review.html): ICO의 의료 MIA·white-box 근거, HIPAA 예외, 환자·외부 연구자·제공기관·규제/인증기관의 역할, 현재 방법의 보조자료와 E/U/P 조건을 대조했다. 사건 배경과 직접 근거를 구분한 2026-09-11 검토이며 실험 계약 변경은 아니다. [Markdown](AUDITOR_FRAMING_REVIEW.md).

[실험별 시간 검토·Spark 4대 계산](runtime_review.html): 기존 D0–D5·K5/K10·공격 비교·제거·효용을 대조한 2026-09-11 계획 검토. 실측과 가정, m=2/m=5 적격 수, MoFit 원형 비용, U 별도 분할·학습의 추가 예산을 구분했다. [계산 JSON](runtime_budget.json), [검증 결과](runtime_verification.json). Spark 사용 시간은 미확정이며 새 학습·공격을 실행하지 않았다.

[CVPR 추진 시 졸업논문 방향](thesis_alignment.html): 기존 프라이버시·출처 주장 검증을 중심에 두고 의료 통합 실증을 강화하는 권장 구조, 세 연구질문, 목차와 공통/추가 실험. 새 감사법 성공이나 CVPR 채택을 졸논의 필수 완료 조건으로 두지 않는다.

[실제 설계 v0.1](design.html): 교차 사진 conditioning 반응 후보, 원래/미사용 사진 시나리오, 강한 비교군, 환자 calibration과 공개 개발 → 공통 K10 평가 순서. [사양 JSON](experiment_design.json)과 [환자 수·비용식 확인](design_feasibility.json)을 함께 기록했다. 아직 새 공격 구현·의료 성능 결과는 아니다.

[연구 목적과 공격의 역할](purpose.html): 원래 보호·보장 환산·재사용/재학습 비용 연구와 공격법을 주기여로 삼는 별도 후보의 관계를 설명한다. 공격 실패로 DP 보장을 인증하거나 변환할 수 없으며, 새 공격법 개발이 자동 확정된 목표는 아니다.

2026-09-10: 기존 관련 67편 + 추가 12편의 개별 검토 기록을 작성했다. 로컬 PDF 선택 절 75편,
공식 본문 일부 3편(A20/A21/B17), 저자 초록만 1편(C02)이다. 모든 페이지 정독·증명 검산·전체
실험 재현을 마친 상태로 표시하지 않는다.

[검색·드롭다운 HTML](index.html) · [전체 기여도 판정](report.html) · [구현 준비](baseline_preparation.html) · [비교군 CSV](comparison_matrix.csv) · [검토 장부 CSV](review_ledger.csv).

[왜 환자 집합 감사법을 우선 검증하는가](rationale.html): 정의·기전 가설·상관과 오탐 계산·선행 반론·DP 경계·성공/중단 조건을 상세히 설명한다. SD-MI(AAAI 2023)와 개인화 의료 Patient Membership Inference(IJCNN 2025)의 추가 원문 선택 절과 메모는 `rationale_sources/`에 있다. 기존 79편 장부의 수와 검토 수준은 유지하고 이 두 편을 별도 보완 기록으로 연결했다. 설명용 수학은 실제 환자 공격 결과가 아니다.

- `registry.json`: 기존 71편과 추가 직접 선행 12편. 기존 탈옥 트렌드 4편은 현재 의료영상 주제 범위 밖으로 표시한다.
- `acquisition.json`: 논문별 원문 확보 결과, 해시·쪽 수. 이 파일의 다운로드 상태는 분석 완료가 아니다.
- `reviews/`: 직접 읽은 내용에 근거한 논문별 방법·가정·실험·한계·비교 준비 메모.
- `code_acquisition.json`: 공식 저장소 commit 또는 공식 supplementary archive와 확보 상태. 코드 실행 여부를 별도 기록한다.
- `pdfs/`, `texts/`, `code_sources/`: 원문 및 대조에 사용한 공개 자료.
- `review_ledger.json`, `comparison_plan.json`: 읽은 범위·역할·출처와 주장별 비교 조건.
- `replays/`: CLiD와 Tracing the Roots 공개 packet의 실제 재계산. 의료 환자 공격 성공이 아니다.
- `evaluate_patient_scores.py`: 고정 점수의 환자 집계와 calibration/test 분리 평가 도구. 새 공격 방법으로 주장하지 않는다.
- `build_review.py`: 개별 Markdown에서 장부·HTML을 다시 생성한다. `test_patient_scores.py`의 핵심 검증 6개 PASS.

읽은 순서는 직접 비교할 공격·학습 방법, 환자 위험·의료영상 선행, 나머지 방법·평가·보장 관련 연구다. 기존 학위논문 학습계약과 private data를 변경하지 않았다. 미해결 원문과 전체 모델 재현의 한계는 종합 보고서와 개별 메모에 남겼다.
