"""Record completed realistic-plan substep A; preserve frozen experiment history."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parents[1]
OUT = THESIS / "code_working/_reports/frozen_residual_pooled_reference_20260916_v1"
REPORT = "TRACK1_POOLED_REFERENCE_RESULTS_20260916.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


analysis, execution, verification = [read(OUT / name) for name in
    ("analysis.json", "execution.json", "independent_verification.json")]
assert verification["complete"] and verification["status"].startswith("PASS")
assert execution["completed"]
summary = dict(report=REPORT, status="completed_independently_verified", current_step=2,
    realistic_plan_substep="5A only", output_directory=str(OUT),
    contract_sha256=sha(OUT/"contract.json"), analysis_sha256=sha(OUT/"analysis.json"),
    verification_sha256=sha(OUT/"independent_verification.json"),
    patients={"public":32,"private":80,"development":40}, families=analysis["families"],
    verification_checks=verification["checks"], execution_seconds=execution["seconds"],
    verification_seconds=verification["seconds"], new_pooled_models=2,
    new_backbone_forward=0,new_backbone_backward=0,generated_images=0,dp_applied=False,
    conclusion="Small descriptive development denoising improvement over public-only; generated private incremental utility remains unestablished.",
    next_substep="5B sampling integration correctness, then separately fixed generation/evaluation",
    next_substep_estimated_work_minutes=[45,90],
    remaining_generation_package_not_completed=True)
updated = datetime.now(timezone.utc).isoformat()
for name in ("research_state.json", "patient_baseline_spec.json"):
    path=ROOT/name
    data=read(path)
    if "completed_track1_pooled_reference_20260916" in data:
        raise RuntimeError("This one-time state update already ran")
    data["pre_pooled_current_result_report_20260916"]=data["current_result_report"]
    data["current_result_report"]=REPORT
    data["updated_utc"]=updated
    data["completed_track1_pooled_reference_20260916"]=summary
    data["next_task"]="Realistic-plan5B: verify fixed-head sampling integration (zero-head base recovery, saved-input offline/online agreement and scheduler semantics); then profile fixed generation."
    data["next_task_output"]="Verified sampling wrapper and numerical evidence; no generated utility claim from tiny denoising differences."
    active=data["active_execution"]
    active["historical_pre_pooled_last_completed"]={k:active.get(k) for k in
        ("last_completed_output","last_completed_report","new_two_track_experiment_status")}
    active.update(status="no_running_execution",last_completed_output=str(OUT),last_completed_report=REPORT,
        additional_execution_contract_fixed=False,next_execution_not_started=True,
        stage2_completion_claim=False,new_two_track_experiment_status="pooled_nonDP_reference_completed_verified",
        completed_track1_pooled_reference=summary,
        latest_planning_status="realistic_plan5A_completed_5B_pending",
        next_substep_estimated_work_minutes=[45,90])
    data["latest_user_execution_scope_20260916"]="Proceed slowly and thoroughly, one substep at a time; section5A completed this turn."
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"updated_utc":updated,"report":REPORT,"states_updated":2}))
