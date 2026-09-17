"""Record a completed design and metadata snapshot, never a model result."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT.parents[1]/'code_working'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report='TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md'
plan=ROOT/'spec_sources/downstream_master_plan_20260917.json'
resource=CODE/'_reports/downstream_master_20260917_v1/resources.json'
r=json.loads(resource.read_text(encoding='utf-8'))
record=dict(report=report,completed_utc=datetime.now(timezone.utc).isoformat(),
    judgment='positive_design_and_development_resources_not_method_efficacy',
    report_sha256=sha(ROOT/report),plan_sha256=sha(plan),resource_sha256=sha(resource),
    public_real=r['public_real'],private_real_diagnostic=r['private_real_diagnostic'],
    development=r['development'],final_ready=False,expert_final_performance_opened=False,
    new_model_forwards=0,new_training=0,new_generation=0,new_DP=0,new_patient_pixel_reads=0,
    original_roles_changed=False,actual_result_unchanged='TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md',
    next_task_eta_minutes=[45,75])
next_task='Implement and bind the downstream runner and fixed manifests, then at most32 separate profile images and100 public-real classifier steps to measure cost and verify integration. Keep expert final and reserved4213 closed. Report actual feasibility/time before the full512-image non-DP development package. No DP or final execution yet.'
for filename in ('research_state.json','patient_baseline_spec.json'):
    p=ROOT/filename; d=json.loads(p.read_text(encoding='utf-8'))
    if 'pre_downstream_master_20260917' in d: raise RuntimeError('Already recorded')
    d['pre_downstream_master_20260917']={k:d.get(k) for k in ('current_planning_report','next_task','next_task_output','step2_active_task','research_question_status','step2_open_items')}
    d['current_step']=2
    d['current_planning_report']=report
    d['next_task_plan_report']=report
    d['next_task']=next_task
    d['next_task_output']='Verified downstream inputs/runner and measured small-profile cost. Full non-DP comparison, medical DP and expert final remain subsequent phases of the master protocol.'
    d['step2_active_task']='Downstream master design and metadata inventory completed; non-DP and DP development precede any expert final performance evaluation.'
    d['step2_open_items']=[
        'Downstream non-DP utility remains unexecuted. Existing192 pooled failure and both evaluator failures are preserved.',
        'Public672/813 contains only6 weak-P positive images; private80/320 has33. Fixed real augmentation and private-real diagnostic are needed to interpret learning efficacy.',
        'Prospective nonfinal selection2027 and method-development2026 groups are disjoint and unconsumed by new pixels/models. Expert532 and reserved4213 remain closed.',
        'Medical-backbone patient-DP and patient-DP LoRA implementation/accounting/cost are pending; do not reuse generic-head weights or nonprotected private calibration.',
        'All final non-DP and DP models, seed banks, contrast families and release scope must be frozen before expert test predictions are inspected.']
    d['research_question_status']='A prospective pneumothorax downstream study is specified using nonfinal weak-label development; no private synthetic benefit, medical DP utility, cost superiority or CVPR contribution established.'
    d['completed_downstream_master_20260917']=record
    active=d['active_execution']
    active['status']='no_running_execution'
    active['latest_design_review']=report
    active['latest_planning_status']='downstream_master_protocol_and_metadata_complete_final_closed'
    active['downstream_master_design']=record
    active['additional_execution_contract_fixed']=False
    for k in ('current_result_report','last_actual_model_result_report'):
        if d[k]!='TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md': raise ValueError(k)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'spec_sources/downstream_master_record_20260917.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
