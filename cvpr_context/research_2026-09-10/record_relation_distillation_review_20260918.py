"""Record a design recommendation without changing completed experiments."""
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
thesis=root.parent.parent
report_name="TRACK1_PATIENT_RELATION_DISTILLATION_COMPARISON_20260918.md"
state_path=root/"research_state.json"
before=json.loads(state_path.read_text(encoding="utf-8-sig"))
state=copy.deepcopy(before)
key="patient_relation_distillation_design_review_20260918"
if key in state:
    raise RuntimeError("Design review already recorded")
tracked=("current_decision_report","current_planning_report","next_task_plan_report",
         "next_task","next_task_output","step2_active_task","research_question_status",
         "next_package_total_work_hours")
state["pre_patient_relation_review_20260918"]={k:before.get(k) for k in tracked}
state["current_decision_report"]=report_name
state["current_planning_report"]=report_name
state["next_task_plan_report"]=report_name
state["next_task"]="Recommend patient-relation risk distillation as the main design, with one bounded A/B/C/D study including held-out-encoder pair-aware reuse and direct-moment controls. No GPU execution started or runtime protocol frozen."
state["next_task_output"]="Implementable comparison contract specifying public encoder/projection, risk/feasibility/count handling and matched budgets; not another open-ended diagnostic queue"
state["step2_active_task"]="Two-design comparison complete; patient-relation distillation recommended over cached condition-contrast diffusion"
state["research_question_status"]="Concrete contribution target is cross-encoder visual reuse of protected patient-centroid contrast risk; efficacy and novelty beyond closest prior are not established"
state["next_package_total_work_hours"]=None
state["updated_utc"]=datetime.now(timezone.utc).isoformat()
state[key]={
    "report":report_name,
    "status":"comparative_recommendation_complete_not_execution_contract",
    "preferred_design":"patient-local centroid-contrast risk distillation",
    "deprioritized_design":"cached condition-contrast diffusion, not experimentally disproved",
    "evidence":"pasted v2, local prior proposal/results, existing roster metadata, scoped primary literature, independent CPU algebra",
    "core_comparison":"point-only vs true relations vs shuffled relations, direct moments and unseen-encoder pair-aware transfer",
    "math_scope":"fixed-feature quadratic centroid risk, not full longitudinal or all-visit-pair risk",
    "new_patient_pixels":0,"new_model_inference":0,"new_training":0,
    "new_synthetic_images":0,"new_dp_releases":0,
    "expert_final_opened":False,"reserved_opened":False,
    "final_ready":False,
    "cpu_check":"spec_sources/relation_distillation_design_review_20260918.json"
}
preserve=[k for k in before if k.startswith("completed_")]
preserve+=["current_result_report","last_actual_model_result_report","last_efficacy_result_report",
           "last_real_generalization_result_report","current_head_branch_closure_20260917",
           "active_execution","steps","current_step","proposed_cached_condition_contrast_20260918"]
for k in preserve:
    assert state[k]==before[k], k
report=root/report_name
for target in re.findall(r"\]\(([^)]+)\)",report.read_text(encoding="utf-8")):
    if not target.startswith(("http://","https://")):
        assert (root/target).is_file(),target
notice=(
    "**2026-09-18 두 설계 비교·주력 권고:** "
    f"[환자 관계 보존 증류와 조건 대조 diffusion 비교]({report_name}). "
    "주력은 환자 내 class-centroid 대조 위험의 DP 증류로 권고하고 직전 diffusion 제안은 후순위로 둔다. "
    "성공 가능성의 증명이 아니라 목적·정보·제거 대조 연결에 근거한 설계 선택이다. "
    "Pairing은 평균이 아닌 관계 2차 모멘트를 바꾸며, noisy moments/count의 일관된 후처리와 "
    "증류에 쓰지 않은 encoder의 pair-aware 재사용을 핵심 비교에 포함한다. "
    "PSG/LGM뿐 아니라 DP-KIP·Dosser·PATH 등 근접 선행도 반영했다. "
    "CPU 배열·기존 명부 metadata만 확인했으며 새 모델/환자 pixel/합성/DP0이다. "
    "권고 완료이지 GPU 실행 계약 동결은 아니다. 실제 결과 Rwide·기존 LoRA 실패, expert532/reserved4213은 보존한다."
)
def prepend(path,text):
    old=path.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] not in old,str(path)
    head,rest=old.split("\n",1)
    path.write_text(head+"\n\n"+text.rstrip()+"\n"+rest,encoding="utf-8")
prepend(root/"RESEARCH_FRAMEWORK.md",notice)
prepend(thesis/"CURRENT_STATUS.md",notice.replace(
    f"]({report_name})",f"](CVPR%20주제%20탐색/research_2026-09-10/{report_name})"))
prepend(thesis/"AGENTS.md",
    "- Latest TWO-DESIGN COMPARISON (2026-09-18): read "+report_name+". "
    "Recommend patient-relation distillation as main design; prior cached diffusion remains an unexecuted lower-priority proposal. "
    "This is a comparative recommendation, not proven efficacy, user execution approval, or a frozen runtime contract. "
    "Preserve patient class-centroid contrast risk, not all longitudinal relations. "
    "Pair shuffle holds mean delta fixed and changes second moments. "
    "Use valid postprocessed noisy counts/moments; direct-moment predictor and unseen-encoder pair-aware reuse are core comparisons. "
    "Pair-blind BCE is secondary, not required to legitimize a pair-aware learner contract. "
    "Closest priors include DP-KIP/Dosser/PATH as well as PSG/LGM/DP-NTK. "
    "New source reading, CPU algebra and old roster metadata only; no pixels/models/GPU/DP releases. "
    "Do not relabel former-selection2027 as public or compare to private80 as a matched-data method study. "
    "Keep all old failures/actual pointers, expert532/reserved4213 closed, stage2/final_ready=false.")
prepend(thesis/"WORKLOG.md",
    "## 153-PATIENT-RELATION-DISTILLATION-COMPARISON (2026-09-18 KST)\n\n"
    "- 사용자가 GPT의 관계 보존 증류 v2와 직전 assistant 조건 대조 diffusion 안을 철저히 비교해 달라고 요청했다.\n"
    "- 주력은 관계 증류로 권고했다. 목적을 환자 class-centroid 대조 위험으로 좁히고 pair-aware cross-encoder 재사용과 direct moments를 핵심 대조로 정리했다. GPU 실행 계약은 아직 동결하지 않았다.\n"
    "- Pair shuffle의 delta 평균 불변/2차 모멘트 변화, centroid와 방문쌍 위험의 차이, PSD/count 후처리, bank rank 한계를 도출했다. 별도 CPU 코드로 검산했다.\n"
    "- 기존 명부를 재집계해 private-origin2027/mixed102, public672/mixed3을 확인했다. 원본 픽셀 접근은 없었다. 감도3, d16좌표459, Gaussian 합계 sigma1.800687, 관계 RMS L2 .217651을 재현했다.\n"
    "- DP-KIP/Dosser/PATH 및 LGM/PSG/DP-NTK/pairwise/의료·시각 distillation primary sources를 대조했다. 전체 신규성이나 성능 성공으로 확대하지 않았다.\n"
    "- 현재 결정·설계 포인터만 새 비교 문서로 갱신하고 모든 completed 실험객체와 실제 Rwide/LoRA 결과 포인터는 동일하게 보존했다. 새 모델·GPU·학습·합성·DP0, expert/reserved 미사용.")
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
record={
    "report":report_name,
    "report_sha256":hashlib.sha256(report.read_bytes()).hexdigest(),
    "prior_completed_records_unchanged":True,
    "actual_result_pointers":{k:state[k] for k in preserve if "result_report" in k},
    "scope":"design recommendation and CPU algebra/old metadata; no new model execution",
    "local_links_exist":True
}
out=root/"spec_sources/relation_distillation_review_record_20260918.json"
assert not out.exists()
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,ensure_ascii=False,indent=2))

