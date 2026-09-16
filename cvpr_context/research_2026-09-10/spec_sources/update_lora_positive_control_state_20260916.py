"""Record completed LoRA diagnostic without changing frozen experiment evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
THESIS=ROOT.parents[1]
OUT=THESIS/'code_working/_reports/frozen_residual_lora_positive_control_20260916_v1'
REPORT='TRACK1_LORA_POSITIVE_CONTROL_RESULTS_20260916.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
replay=read(OUT/'execution_replay.json');bridge=read(OUT/'execution_bridge.json')
v=read(OUT/'independent_verification_v2.json')
manifests=read(OUT/'manifest_replay.json')+read(OUT/'manifest_bridge.json')
if not(v['complete'] and v['status'].startswith('PASS') and replay['completed'] and bridge['completed'] and len(manifests)==12):
    raise RuntimeError('Completed verified twelve-image packet required')
summary=dict(report=REPORT,protocol='TRACK1_LORA_POSITIVE_CONTROL_PROTOCOL_20260916.md',
    status='completed_positive_path_diagnostic_generated_quality_inadequate',current_step=2,
    output_directory=str(OUT),contract_sha256=sha(OUT/'contract.json'),
    amendment_sha256=sha(OUT/'verification_amendment_v2.json'),
    verifier_result_sha256=sha(OUT/'independent_verification_v2.json'),
    research_judgment='Favorable diagnostic: current conditions permit basic CXR form with adapted LoRA. Final images have clear artifacts/distortion; clinical utility and small-head efficacy remain unestablished.',
    historical_pngs_exact=2,pipeline_manual_complete_values_exact=True,
    independent_final_checks=v['checks'],DDIM_transitions=v['DDIM_transitions_checked'],
    UNet_calls=360,UNet_batch_examples=540,backward=0,VAE_decodes=12,
    runner_seconds=replay['seconds']+bridge['seconds'],trajectory_seconds=sum(x['seconds'] for x in manifests),
    peak_allocated_gpu_bytes=max(x['peak_memory_bytes'] for x in manifests),
    original_verifier_failure_preserved=True,timestep_storage_correction_only=True,
    new_training=False,new_small_head_applied=False,new_quality_metrics=False,
    public_backbone_valid=False,small_head_efficacy_established=False,DP_claim=False,
    causal_single_failure_source_established=False,clinical_quality_established=False,
    next_work='Fix public medical backbone acquisition/training data boundary and fresh-head comparison before new training.',
    next_design_work_minutes=[30,45],automatic_96image_expansion=False)
visual=dict(utc=datetime.now(timezone.utc).isoformat(),reviewers=['root','u_inference_audit'],
    scope='Both reviewers inspected all twelve grid cells and both final images; coarse morphology only, no clinician assessment.',
    original_replay='Two historical PNGs exact and basic frontal chest-radiograph form reproduced.',
    final_normal='Chest-like structure with repeated thin lines and distorted boundaries.',
    final_effusion='Chest-like ribs/spine with severe superposition/rotation/shape distortion.',
    quality_judgment='Not a clinical-quality or disease-conditioning success.',
    image_sha256={x['image_path']:x['image_sha256'] for x in manifests},
    grid_sha256=sha(OUT/'comparison_grid.png'),no_cherry_picked_images=True)
with (OUT/'visual_review.json').open('x',encoding='utf-8') as f:json.dump(visual,f,indent=2);f.write('\n')
for name in ('research_state.json','patient_baseline_spec.json'):
    path=ROOT/name;s=read(path)
    if 'completed_track1_lora_positive_control_20260916' in s:raise RuntimeError('One-time state update already ran')
    s['pre_lora_control_current_result_report_20260916']=s['current_result_report']
    s['current_result_report']=REPORT
    s['completed_track1_lora_positive_control_20260916']=summary
    s['updated_utc']=datetime.now(timezone.utc).isoformat()
    s['next_task']='Fix a public-only medical-backbone data/training boundary and fresh-head comparison; do not relabel diagnostic M1 as public or reuse its old generic-base W.'
    s['next_task_output']='One bounded public-backbone acquisition/training specification with quality criterion, patient roles, compute estimate and independent evaluation boundary.'
    s['research_question_status']='Adapted LoRA supports basic CXR form under current sampling inputs, with serious artifacts. Public medical backbone plus fresh patient-private small head remains an unvalidated application candidate.'
    s['step2_active_task']='Specify a valid public medical generation starting point before further small-head privacy/utility comparisons.'
    a=s['active_execution']
    a['historical_pre_lora_last_completed']={k:a.get(k) for k in ('last_completed_output','last_completed_report','new_two_track_experiment_status')}
    a.update(status='no_running_execution',last_completed_output=str(OUT),last_completed_report=REPORT,
        additional_execution_contract_fixed=False,next_execution_not_started=True,stage2_completion_claim=False,
        new_two_track_experiment_status='lora_path_diagnostic_positive_quality_inadequate',
        completed_track1_lora_positive_control=summary,
        latest_planning_status='public_medical_backbone_boundary_and_retraining_design_next',
        next_substep_estimated_work_minutes=[30,45],automatic_96image_expansion=False)
    if 'current_sampling_scope' in a:a['historical_pre_lora_sampling_scope']=a.pop('current_sampling_scope')
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(states_updated=2,report=REPORT,checks=v['checks'],runner_seconds=summary['runner_seconds'])))
