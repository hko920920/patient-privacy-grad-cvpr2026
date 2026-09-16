"""Record integration correctness and negative basic-generation applicability."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
THESIS=ROOT.parents[1]
OUT=THESIS/"code_working/_reports/frozen_residual_sampling_integration_20260916_v1"
REPORT="TRACK1_SAMPLING_INTEGRATION_RESULTS_20260916.md"


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


e,v,m,p=[read(OUT/n) for n in ("execution.json","independent_verification.json","manifest.json","preview/manifest.json")]
if not(e["completed"] and v["complete"] and v["status"].startswith("PASS") and p["images_count"]==6):
    raise RuntimeError("Completed and verified packet with six previews required")
summary=dict(report=REPORT,protocol="TRACK1_SAMPLING_INTEGRATION_PROTOCOL_20260916.md",
    status="completed_verified_integration_negative_generation_applicability",current_step=2,
    output_directory=str(OUT),contract_sha256=sha(OUT/"contract.json"),
    verifier_result_sha256=sha(OUT/"independent_verification.json"),
    preview_manifest_sha256=sha(OUT/"preview/manifest.json"),
    implementation_judgment="PASS",research_judgment="Poor in the current fixed two-case configuration: all six previews are grossly non-CXR.",
    independent_checks=v["checks"],DDIM_transitions=300,original_public_witnesses_exact=4,
    zero_and_restored_full_trajectories_exact=True,UNet_forward=e["forward_calls"],backward=0,VAE_decodes=6,
    execution_seconds=e["seconds"],verification_seconds=v["seconds"],
    trajectory_seconds=sum(x["seconds"] for x in m["trajectories"]),
    peak_allocated_gpu_bytes=max(x["peak_memory_bytes"] for x in m["trajectories"]),
    generated_images=6,generated_quality_metrics=False,new_DP_training=False,
    cohort_evaluation="Public witness reconstruction only; no final calibration/test evaluation",
    decision="Pause96-image expansion and solver variants; verify existing CXR-form LoRA positive control under matched generation path.",
    causal_attribution_to_head_or_guidance_established=False,
    private_incremental_generation_value_established=False,
    positive_control_scope="Existing M1/M2 used private-role training data; diagnostic only, not a patient-DP-safe public backbone.",
    next_work_minutes=[30,60])
for name in ("research_state.json","patient_baseline_spec.json"):
    path=ROOT/name;s=read(path)
    if "completed_track1_sampling_integration_20260916" in s:
        raise RuntimeError("One-time state update already ran")
    s["pre_sampling_current_result_report_20260916"]=s["current_result_report"]
    s["current_result_report"]=REPORT
    s["completed_track1_sampling_integration_20260916"]=summary
    s["updated_utc"]=datetime.now(timezone.utc).isoformat()
    s["next_task"]="Use the existing step1000 LoRA CXR-form positive control to check the bridge between its known generation path and current sampling conditions; no automatic96-image expansion or solver search."
    s["next_task_output"]="Matched-input positive-control specification and generation-path evidence, keeping diagnostic private-trained LoRA separate from a DP-valid public backbone."
    s["research_question_status"]="Application question remains open; current general-base plus small-head fixed preview failed basic CXR-form applicability, despite correct integration."
    s["step2_active_task"]="Re-establish a valid generation starting point and sampling control before extending privacy/utility comparisons."
    s["result_reporting_policy"]="Lead with favorable/unclear/poor research judgment, then implementation evidence and next decision."
    a=s["active_execution"]
    a["historical_pre_sampling_last_completed"]={k:a.get(k) for k in ("last_completed_output","last_completed_report","new_two_track_experiment_status")}
    a.update(status="no_running_execution",last_completed_output=str(OUT),last_completed_report=REPORT,
        additional_execution_contract_fixed=False,next_execution_not_started=True,
        stage2_completion_claim=False,new_two_track_experiment_status="sampling_verified_but_current_generation_bad",
        completed_track1_sampling_integration=summary,
        latest_planning_status="96image_expansion_paused_existing_positive_control_bridge_next",
        next_substep_estimated_work_minutes=[30,60],automatic_96image_expansion=False)
    if "proposed_next_package_total_hours" in a:
        a["historical_pre_sampling_package_total_hours"]=a.pop("proposed_next_package_total_hours")
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"states_updated":2,"report":REPORT,"research_judgment":"poor_current_generation","verification_checks":v["checks"]}))
