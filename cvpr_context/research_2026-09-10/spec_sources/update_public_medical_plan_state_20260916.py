"""Publish a completed planning step while preserving the latest actual result."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib
R=Path(__file__).resolve().parents[1]
P=R/'spec_sources/public_medical_backbone_plan_20260916_v1'
REPORT='TRACK1_PUBLIC_MEDICAL_BACKBONE_PLAN_20260916.md'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=read(P/'execution_spec.json');verify=read(P/'execution_spec_verification.json')
if not verify['status'].startswith('PASS') or spec['new_GPU_runs']!=0:raise RuntimeError('Planning verification required')
summary=dict(report=REPORT,execution_spec=str(P/'execution_spec.json'),execution_spec_sha256=sha(P/'execution_spec.json'),
    status='completed_design_and_cpu_verification_no_training',current_step=2,track=1,
    research_judgment='Favorable readiness: a separate local public cohort is available; new backbone/head efficacy remains untested.',
    training_patients=640,training_images=749,selection_patients=128,confirmation_patients=128,reserve_patients=9,
    planned_checkpoint_steps=[749,1498],planned_committed_exposures=[4,8],
    next_package_minutes=[50,85],pure_training_minutes=[10,16],planned_max_generation_images=52,
    actual_new_GPU_runs=0,actual_new_training=0,actual_generated_images=0,
    current_actual_result_unchanged=True,independent_plan_verification_sha256=sha(P/'execution_spec_verification.json'),
    P_policy='Same data-independent seeded projection can be retained; fresh backbone-dependent statistics and W required.',
    first_gate_unit='generated inputs; reserved real-patient reference sets not evaluated yet')
for name in ('research_state.json','patient_baseline_spec.json'):
    p=R/name;s=read(p)
    if 'completed_public_medical_backbone_plan_20260916' in s:raise RuntimeError('One-time update already applied')
    s['completed_public_medical_backbone_plan_20260916']=summary
    s['current_planning_report']=REPORT
    s['next_task']='Implement and run the single fixed public-only LoRA pilot from fresh SD2.1 on640patients/749images: E4/E8 then prescribed selection and once-only new-input confirmation.'
    s['next_task_output']='New isolated public cache/trainer with frozen runtime code, one training trace/checkpoints, all prescribed generated inputs, integrity evidence and an explicit coarse-form quality decision.'
    s['step2_active_task']='Execute the now-fixed public-backbone feasibility package; small private head and DP solver expansion remain later conditional steps.'
    s['research_question_status']='Public medical backbone plus fresh small patient-private adaptation remains unvalidated; public data allocation and exposure/selection specification are now fixed.'
    s['updated_utc']=datetime.now(timezone.utc).isoformat()
    a=s['active_execution']
    a['historical_pre_public_plan_next_task']=a.get('latest_planning_status')
    a.update(status='no_running_execution',latest_design_review=REPORT,
        latest_planning_status='public_backbone_manifest_exposure_selection_plan_complete_training_not_started',
        additional_execution_contract_fixed=False,next_execution_not_started=True,
        next_substep_estimated_work_minutes=[50,85],completed_public_medical_backbone_plan=summary,
        automatic_96image_expansion=False,stage2_completion_claim=False)
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(states_updated=2,planning_report=REPORT,actual_result_unchanged=True)))
