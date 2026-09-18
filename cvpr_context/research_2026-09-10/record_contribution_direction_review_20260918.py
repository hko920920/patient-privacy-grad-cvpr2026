from __future__ import annotations
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parent
thesis = root.parent.parent
report_name = "TRACK1_CONTRIBUTION_DIRECTION_AUDIT_20260918.md"
report = root / report_name
state_path = root / "research_state.json"
before = json.loads(state_path.read_text(encoding="utf-8-sig"))
state = copy.deepcopy(before)
stamp = datetime.now(timezone.utc).isoformat()
tracked = [
    "current_decision_report", "current_planning_report", "next_task_plan_report",
    "next_task", "next_task_output", "step2_active_task", "research_question_status",
    "next_package_total_work_hours"
]
state["pre_contribution_direction_audit_20260918"] = {
    key: before.get(key) for key in tracked
}
state["current_decision_report"] = report_name
state["current_planning_report"] = report_name
state["next_task_plan_report"] = report_name
state["step2_active_task"] = "End-to-end contribution-direction review complete; valid diagnostics do not establish a convergent paper contribution"
state["research_question_status"] = "Protection-efficiency question remains reasonable; original SSP/head and fixed LoRA have no validated private synthetic advantage; support diagnostic is positive but not protected generation evidence"
state["next_task"] = "If continuing, connect one contribution hypothesis to explicit public/private access, useful synthetic transfer and whole-cost comparisons. No automatic next diagnostic or experiment is selected by the Rwide result."
state["next_task_output"] = "One bounded matched-access research design whose outcomes decide a contribution claim; not another open-ended troubleshooting queue"
state["next_package_total_work_hours"] = None
state["updated_utc"] = stamp
state["completed_contribution_direction_audit_20260918"] = {
    "report": report_name,
    "status": "review_complete_contribution_not_established",
    "judgment": "Current workflow is diagnostically useful but not demonstrated to converge to a CVPR contribution",
    "scope": "Historical objectives, key protocols/results/current state, saved Rwide metrics, scoped primary-literature check",
    "new_model_runs": 0, "new_training": 0, "new_generation": 0,
    "new_patient_pixel_access": 0, "new_DP": 0,
    "old_head_failure_preserved": True, "old_LoRA_failure_preserved": True,
    "initial_generic_head_DP_acknowledged": True,
    "expert_final_opened": False, "reserved_opened": False,
    "new_algorithm_required": False, "topic_switch_selected": False,
    "next_experiment_selected": False, "final_ready": False
}
# No experiment records, outcomes, or last-actual/efficacy pointers may change.
preserved_keys = [k for k in before if k.startswith("completed_")]
preserved_keys += [
    "current_result_report", "last_actual_model_result_report",
    "last_efficacy_result_report", "last_real_generalization_result_report",
    "current_head_branch_closure_20260917", "active_execution", "steps",
    "current_step"
]
for key in preserved_keys:
    assert state[key] == before[key], key

def prepend_after_heading(path: Path, text: str):
    old = path.read_text(encoding="utf-8-sig")
    first, rest = old.split("\n", 1)
    assert text.split("\n")[0] not in old, str(path)
    path.write_text(first + "\n\n" + text.rstrip() + "\n" + rest, encoding="utf-8")

# Check only this review's references, not all historical experiment artifacts.
references = []
for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
    if re.match(r"https?://", target):
        references.append({"target": target, "kind": "web_reference"})
    else:
        resolved = (root / unquote(target.split("#")[0])).resolve()
        assert resolved.is_file(), target
        references.append({"target": target, "kind": "local_reference", "exists": True})
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

notice = (
    "**2026-09-18 全過程 기여 방향 재검토:** "
    f"[연구 진단과 논문 기여의 연결]({report_name}). "
    "보호 설계·효율의 문제 가치는 유지하지만 현재 진단 연쇄가 CVPR 핵심 기여에 수렴했다는 근거는 부족하다. "
    "초기 generic head DP에서 SSP가 same-head DP-SGD보다 약했던 결과도 함께 판단한다. "
    "Rwide는 실자료 범위의 양성 진단이며 private synthetic·보호 효용·총비용 우위가 아니다. "
    "새 알고리즘 또는 주제 변경을 자동 선택하지 않고, 자료 접근·효용·비용이 결속된 한 기여 가설로 다음 연구를 판단한다. "
    "이번은 문헌·저장기록 검토만 완료했고 새 모델/영상/DP0이다. "
    "실제 최신 모델 결과는 Rwide, 사적 생성 효용 결과는 LoRA 실패로 유지하며 expert/reserved는 닫혀 있다."
).replace("全過程", "전 과정")
prepend_after_heading(root / "RESEARCH_FRAMEWORK.md", notice)
prepend_after_heading(thesis / "CURRENT_STATUS.md", notice.replace(
    f"]({report_name})", f"](CVPR%20주제%20탐색/research_2026-09-10/{report_name})"))
prepend_after_heading(thesis / "AGENTS.md",
    "- Latest CONTRIBUTION-DIRECTION REVIEW (2026-09-18): read "
    + report_name + ". Review completed with no model/pixel/DP execution. "
    "Valid individual diagnostics do not establish a convergent CVPR contribution. "
    "Preserve initial generic-head SSP-vs-cached-DP-SGD negative result as well as medical head/LoRA failures. "
    "Rwide remains a positive NONDP real-support diagnostic, not protected generation evidence or a new public baseline. "
    "Current decision report is the review; last actual result remains Rwide and last private-generation efficacy remains LoRA failure. "
    "No new algorithm, topic switch, cohort or next experiment was selected. "
    "Known-principle application contributions remain allowed; novel DP math and exhaustive causal diagnosis are not prerequisites. "
    "Future work should link explicit data access, meaningful private synthetic utility and total cost to one contribution hypothesis; "
    "do not automatically extend the diagnostic queue. Expert532/reserved4213 remain closed; stage2/final_ready=false.")
prepend_after_heading(thesis / "WORKLOG.md",
    "## 151-CONTRIBUTION-DIRECTION-AUDIT (2026-09-18 KST)\n\n"
    "- 사용자는 처음부터 전체 연구가 기여를 만드는 방향인지 재검토하라고 요청했다. "
    "원래 보호 목표, 공격 탐색, 두 v0, frozen residual/초기DP, 의료 backbone/head, 측정·downstream·LoRA·일반화·Rwide 경로를 추적했다.\n"
    "- 판단: 개별 진단은 유효하지만 현재 흐름을 CVPR 기여에 수렴 중이라고 평가할 근거는 부족하다. "
    "초기 SSP의 same-head DP-SGD 대비 음성 결과와 전체비용 경계를 재포함했다. "
    "Rwide 실자료 범위 개선은 private synthetic/DP 효용을 구제하지 않는다.\n"
    "- DP-LoRA/DP-LDM, DP-MEPF, Private Evolution, PDA-DPMD, DPImageBench와 사용자단위 보호의 근접 범위를 공식·저자자료로 확인했다. "
    "원리 중복만으로 응용 기여를 배제하지 않고, 실제 추가 가치·동일 접근·보호·총비용을 함께 요구한다.\n"
    "- 새 알고리즘/주제 변경/추가 GPU를 선택하지 않았다. 기존 실패·실험객체·실제 결과 포인터와 expert/reserved를 보존했다. "
    "새 문서와 상태의 의사결정 필드만 갱신했다. 새 학습/추론/생성/환자영상접근/DP0.\n"
    "- 전체 원본·checkpoint 재실행은 없고, 이번 확인은 주요 기록과 최신 Rwide result, 문헌 및 새 문서 연결 범위다. "
    "검산 항목 수를 새로운 성능 증거로 추가하지 않았다.")

record = {
    "schema": "contribution-direction-review/v1", "created_utc": stamp,
    "report": report_name, "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "review_only": True, "no_new_experiment": True,
    "preserved_actual_result_pointers": {
        k: state[k] for k in [
            "current_result_report", "last_actual_model_result_report",
            "last_efficacy_result_report", "last_real_generalization_result_report"
        ]
    },
    "prior_experiment_records_unchanged": True,
    "local_links_resolved": True,
    "references": references,
    "scope_limitation": "No new image decoding, checkpoint inference, performance bootstrap or global historical hash audit"
}
(root / "spec_sources/contribution_direction_review_20260918.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
assert json.loads(state_path.read_text(encoding="utf-8")) == state
print(json.dumps({
    "review": report_name,
    "last_actual_result": state["last_actual_model_result_report"],
    "last_private_generation_efficacy": state["last_efficacy_result_report"],
    "new_experiment": False, "old_experiment_records_unchanged": True,
    "new_review_local_links_exist": True
}, ensure_ascii=False))

