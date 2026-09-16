# PP-Mark·ICLR jailbreak·CVPR U 설계 사례 실제 기록 대조

2026-09-14. 사용자가 실제 사례를 읽지 않고 이해했다고 답한 문제를 지적한 뒤 수행한 검토다. **직전 답변 때는 PP-Mark와 ICLR 폴더를 직접 확인하지 않았다.** 이번에는 원래 목표·방법 기록·후속 결정과 아래 결과 파일을 직접 열었다. 모든 파일·모든 논문을 정독하거나 과거 실험을 재실행한 검토는 아니다.

## 1. PP-Mark: 설계로 통제하는 관계와 실제 성과가 있다

실제 경로: `Documents/PP-Mark-v0.4/PP-Mark`. 별도 `Documents/GitHub/PP-Mark` 사본도 확인했지만 두 저장소의 git 이력을 동일시하지 않았다.

- v0.4 사본의 2025-12-01 커밋 `b5d4f7b`에 있는 초기 구현 계획을 직접 읽었다. lookup/addition 중심 회로, FFT 회피, 부분 샘플링으로 비용을 줄이겠다는 구체적 계산상 이유가 있다. 당시 Halo2 stub과 skipped 공격시험도 명시돼 있으므로 초기 목표와 달성 결과를 구별한다. 현재 파일은 SP1로 갱신됐다.
- `docs/architecture.md:48–64`의 현재 Attest 조건은 검출 점수 외에 신뢰된 producer commitment, 정확한 이미지/공개 statement와 증명을 요구한다. 특정 공개 검출 점수를 최적화하거나 다른 이미지의 신호를 옮기는 행동만으로 이 조건을 충족하지 못하도록 프로토콜에 관계를 넣었다. 이는 학습 이후 유용한 신호가 저절로 남아 있을 것이라는 기대보다 설계자가 직접 통제하는 부분이 크다.
- `docs/neurips_v2_execution_plan.md:3–15`에는 2026-04-21 opened subset의 embedding relation을 회로에서 강제하도록 확장하기로 한 기록이 있다. 현재 P1–P4 전체가 초기부터 완성됐다고 주장하지 않는다.
- 실제 regeneration 원본 `outputs/attacks/muller_forgery_report_prep/regen_zhao_full100/pp_mark/{weak,med,strong}/pos/summary.json`을 열어 각각 count=100, soft_pass_rate=.98을 확인했다. 같은 폴더 집계표는 Tree-Ring 95/93/93%, Stable Signature 4/2/3%를 기록한다. 이 수치는 해당 조건의 score 검출 성과다.
- 8월2일 증거 ZIP 안의 `v6_a100_cuda_long20.json`, `v6_a100_cuda_long20_r2.json`을 직접 읽었다. 각각 20회, SP1 6.3.1/CUDA, streaming-full-trace, N1000, execute_only=false다. 첫 캠페인 proving 중앙값1.62465초/verification .18248초, setup33.322초 별도. 두 번째 proving 중앙값1.71177초다. 실제 후속 proof 동작 근거이므로 7월의 미완료 기록만으로 현재 상태를 판단하면 틀린다.

따라서 이 사례에는 **구체적인 실패 경로 → 차단할 관계/계산 구조 → 구현 → 일부 비교에서 확인된 성과**가 있다. 초기의 모든 목표나 모든 SOTA 지표가 필연적으로 달성됐다는 뜻은 아니다.

검토에서 확인한 중요한 범위: regeneration JSON의 accept_pass_total=0, accept_pass_rate=null이므로 .98을 변환 이미지의 Attest 성공률로 바꾸지 않는다. `scripts/experiment_sig_vs_zkp.py:464,475`의 far_ppmark_accept=0.0은 상수다. 이 표의 0을 100건의 실제 proof 검증 측정으로 말하지 않는다. 실제 score 측정과 설계상 선언값, 별도의 실제 proving 캠페인을 분리했다. 현재 논문의 모든 암호학적 주장을 독립 재증명한 것은 아니다.

## 2. ICLR: 가설은 있었지만 중요한 전제와 추가 가치가 확보되지 않았다

실제 경로: `Documents/jailbreak iclr 2027`.

원래 `docs/PROJECT_CHARTER.md`의 질문은 원래 요청을 유지하면서, 어떤 공격 추가 문구를 중립화하면 거부가 회복되는지 찾는 것이다. H1 국소화, H2 규모 이질성, H3 연산자/seed/judge 강건성, H4 질의 효율, H5 기권을 명시했다. 새 jailbreak 생성 방법은 원래 범위 밖이다. 따라서 단순히 '새 공격법이 실패했다'고 요약하면 이 사례를 잘못 이해한 것이다.

- `docs/PAPER_SCOPE_DECISION_V2.md:56–66`은 모든 최소 회복 집합·고차 상호작용·대조·연산자 일치 등의 결합과 반복 구조가 있으면 독립 경험 논문에 충분하다고 먼저 판단했다. 그 구조가 강한 단순 대안보다 어떤 중요한 정보를 주는지는 아직 입증되지 않았다.
- D3 원본 `data/natural_language_localization/d3_exact_topology_v1/result.safe.json`을 직접 열었다. 새 target 생성1,807회, 능력 대조324회, 보고 가능11/12, 비단순 구조10개가 있지만 d3_core_gate_pass=false다. 긍정 관측 자체가 전혀 없었던 것은 아니다. 연산자 일치 등 핵심 기준을 충족하지 못했다.
- 후속 `docs/PA_POST_FOCUS_CONTRIBUTION_DECISION_2026-09-08_V1.md:20–24,89–98`은 실제 판단 손실·단순 재검사 이상의 가치·같은 비용 개선이 확보되지 않았다고 명시한다. 전이 실패65/128 중60개가 형식 오류였으며, 그 결과가 그대로 원래 보안 기여가 되지는 않았다.
- `docs/PA_COMMON_ORDER_PREDICTION_EXECUTION_2026-09-08_V1.md:62–112`의 실제 수치 재분석은 learned common order Brier .05572, 단순 pooling .01710이다(낮을수록 좋음). 학습한 순서와 pooled 순서는 관측값이 모두0인 두 열의 순서만 달랐다. 관측된 구조가 더 복잡한 방법의 유용한 추가 정보가 되지 못한 구체적 사례다.
- lazy-panel 기록은 105개 기존 judge 호출이 논리적으로 생략 가능하다고 계산했지만, 강한 단순 short-circuit/층화 추론보다 독립적인 방법 기여를 확보하지 못했다고 스스로 결론 냈다. 이는 실제 새 walltime 개선 실험이 아니다.
- 마지막 `docs/TASK_FIDELITY_LAST_CHANCE_EXECUTION_2026-09-09_V1.md`는 구현 테스트316 PASS 뒤 실제32회에서 원문4/16·완전 보존 대조2/16, 적격 쌍0/8을 기록한다. 원본 `artifacts/task_fidelity_last_chance_v1/run-20260909-01/result.json`도 ORIGINAL6/OTHER22/UNPARSEABLE4, baseline_gate.pass=false, total_model_calls=32다. 계획 행48+192개를 실제 실행으로 세지 않는다. 이후240회는 미실행이며 프로젝트 투자도 최종 중단됐다.

`docs/PA_DESIGN_ACCOUNTABILITY_AUDIT_2026-09-09_V1.md`는 다른 요청의 구조적 단서를 옮긴 뒤 전제를 확인하기 전에 전체 runner를 만들었던 순서 문제도 이미 인정한다. 기록에는 정교한 가설·검증 절차·부정적 결과 보존이 있으나, 그것만으로 해당 후보에 투자할 과학적 근거가 강해지는 것은 아니다.

## 3. 현재 CVPR U와 연결되는 실제 문제

`EXPERIMENT_DESIGN.md:48–79`와 `U_EXECUTION_PLAN.md:56–67`을 직접 다시 읽었다. 후보 R은 같은 환자의 사진 간 conditioning 반응 전이가 미세조정 참여와 관련되며, 유사 타환자 차감 후 기존 집계 이상의 정보가 남는다는 연쇄 가설이다. 문서에도 가설이지 관측 사실이 아니라고 적혀 있다.

검출 가능한 학습 참여 흔적, 그 흔적의 다른 사진 전이, 선택한 탐색 공간의 적합성, 참조 차감의 유용성, 강한 집계 대비 추가 정보가 아직 입증되지 않았다. 구현과 수치 검사를 통과해도 이 전제를 보충하지 못한다. 현재 SecMI 결과와 R 결과는 확대 근거를 주지 못했고 보류 상태를 유지한다.

## 4. 이번 답변에서 바로잡을 판단

사용자의 요구는 계획서·테스트·기록의 양을 늘리라는 것이 아니다. **좋은 결과를 기대하는 이유가 기존 방법의 실제 한계와 연결돼 있고, 그 이유를 검사하는 핵심 결과가 먼저 있어야 한다는 요구**다.

PP-Mark는 설계자가 통제할 차단 관계와 일부 실측 성과가 연결돼 있다. ICLR/CVPR 후보들은 의미 있는 현상의 존재와 강한 대안 대비 추가 가치를 아직 확인해야 했는데, 실행 준비와 논문 가능성을 너무 가깝게 설명했다. '논리적으로 정의할 수 있고 평가할 수 있음'에서 '투자할 만한 강한 연구 후보임'으로 넘어갈 근거가 부족했다.

이번 기록 대조로 새 연구 후보가 만들어졌다고 주장하지 않는다. 새로운 모델 실행·수정·튜닝은 하지 않았다. 현재 CVPR 확대 보류와 ICLR의 기존 최종 중단을 변경하지 않았다.

## 직접 열어 본 핵심 출처 범위

- PP-Mark: 과거 커밋의 구현 계획, 현재 구현 계획, architecture, method_draft_log, related_works_log, experiment_set_plan, neurips_v2_execution_plan; paper/example_paper.tex의 도입·관련연구·방법·정리·선택 결과 절; 7월29일 후속 기록의 8월 측정/수정 항목; regeneration 세 JSON/집계표; signature 비교 코드 해당 구간; 8월 ZIP의 두 timing JSON.
- ICLR: PROJECT_CHARTER, FORMAL_PROBLEM 앞부분, PAPER_SCOPE_DECISION_V2 해당 절, 설계 책임 감사와 마지막 실행 기록, lazy-panel 가설 기록, common-order 실행 앞부분, 전략 reset 기록의 핵심 절, post-focus 판단 해당 절; D3와 마지막32회 결과 JSON. 별도 검토자는 D3·C1N·focus의 후속 문서/SAFE 집계도 대조했다.
- CVPR: 원 설계/U 실행 계획의 핵심 절과 이미 수행한 U 재검증·SecMI 결과. 관련연구 수십 편 전체를 이번에 새로 정독한 것으로 표시하지 않는다.

검토 소요 약12분. 새 GPU·모델·공격 실행0. 기존 결과 파일은 수정하지 않았다.
