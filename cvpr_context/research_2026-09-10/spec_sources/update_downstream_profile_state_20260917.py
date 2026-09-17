"""Record mixed bounded-profile outcome without upgrading the failed classifier gate."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT.parents[1]/'code_working'
OUT=CODE/'_reports/downstream_profile_20260917_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

report='TRACK1_DOWNSTREAM_PROFILE_RESULTS_20260917.md'
v=read(OUT/'verification_v2.json');g=read(OUT/'generation.json');repair=read(OUT/'classifier_repair_v2/result.json')
if v['original_classifier_gate']!='FAIL' or repair['total_optimizer_updates']!=100:raise RuntimeError('Outcome mismatch')
paths={
    'protocol':ROOT/'TRACK1_DOWNSTREAM_PROFILE_PROTOCOL_20260917.md','report':ROOT/report,
    'execution_contract':OUT/'contract.json','data':OUT/'data.json','generation':OUT/'generation.json',
    'generation_manifest':OUT/'generation_manifest.json','grid':OUT/'profile_grid.png',
    'original_classifier':OUT/'classifier.json','classifier_failure':OUT/'classifier_verification_failure.json',
    'repair_contract':OUT/'classifier_repair_v2/contract.json','repair_result':OUT/'classifier_repair_v2/result.json',
    'verification':OUT/'verification_v2.json','raw_verified_overlap':OUT/'raw_verified_overlap.json',
    'sampler_semantics':OUT/'sampler_semantic_checks.json','consumption':OUT/'consumption_summary.json'}
record=dict(report=report,completed_utc=datetime.now(timezone.utc).isoformat(),
    judgment='mixed_generator_pass_original_classifier_fail_limited_repair',profile_fully_passed=False,
    raw_images_verified=11277,raw_patients=4805,generated_distinct_images=28,generation_decodes=31,
    UNet_calls=931,generator_backward=0,original_invalid_classifier_updates=98,corrected_classifier_updates=2,
    total_classifier_updates=100,corrected_CPU_batch_plans=49,corrected_training_replay='S1 single step twice, exact',
    corrected_complete_seven_arm_training_replay=False,final_ready=False,
    expert_final_pixels=0,expert_final_predictions=0,reserved_pixels=0,reserved_predictions=0,
    development_pixel_status='All4053 development patients decoded;128 development images cost-only predictions saved, not used for AUROC/AP or selection.',
    AUROC_AP_computed=False,private_incremental_utility_established=False,new_DP=0,
    full512_generation_executed=False,full21classifier_runs_executed=False,
    previous_efficacy_result='TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md',
    original_roles_changed=False,remote_main_checked=False,remote_push=False,
    next_task_eta_minutes=[10,15],full_nonDP_package_provisional_eta_minutes=[90,150],
    source_paths={k:str(p) for k,p in paths.items()},source_sha256={k:sha(p) for k,p in paths.items()})
next_task='Separate corrected seven-arm runtime replay: reuse existing profile bank,7 arms x1 update x2 repeats=14 updates, bind exact real/synthetic namespaces and verify independently. Prior profile100-update cap stays closed. No new generation, AUROC/AP, DP, expert-final or reserved access. Only after this passes execute the master512-image/21-run non-DP development package.'
for filename in ('research_state.json','patient_baseline_spec.json'):
    p=ROOT/filename;s=read(p)
    if 'completed_downstream_profile_20260917' in s:raise RuntimeError('Already recorded')
    s['pre_downstream_profile_actual_result']=s['current_result_report']
    s['current_step']=2;s['current_result_report']=report;s['last_actual_model_result_report']=report
    s['last_efficacy_result_report']='TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md'
    s['current_planning_report']='TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md'
    s['next_task_plan_report']=report;s['next_task']=next_task
    s['next_task_output']='Corrected seven-arm minimal runtime replay and valid source-composition verification; full non-DP efficacy study remains subsequent.'
    s['step2_active_task']='Bounded downstream profile complete with mixed result; generator passes, original classifier source composition failed and limited repair passed. Corrected seven-arm runtime check remains.'
    s['research_question_status']='Private synthetic downstream utility remains unmeasured. Engineering profile is mixed, not a method efficacy pass; final_ready=false.'
    s['step2_open_items']=[
        'Original98 classifier updates invalid for requested real/synthetic comparison; source namespace repaired and two S1 updates exact, but corrected full seven-arm runtime replay not done.',
        'Full512-image synthetic bank, public R1 calibration,21 classifier runs and S2/S3 versus S1/R1 development metrics remain unexecuted.',
        'Development4053 patients are now consumed for preprocessing;128 development images had cost-only inference. Final532 and reserved4213 remain closed.',
        'Medical patient-DP/accounting and DP-LoRA pending non-DP development utility gate.',
        'Stronger real augmentation, second ImageNet architecture and broader negative prompts/independent bank remain development robustness work if first pilot is positive.',
        'Expert final may be opened only after all final non-DP/DP models and analysis are frozen; old192 and both evaluator failures remain unchanged.']
    s['completed_downstream_profile_20260917']=record;s['updated_utc']=record['completed_utc']
    active=s['active_execution'];active['status']='no_running_execution'
    active['last_completed_report']=report;active['last_completed_output']=str(OUT)
    active['downstream_profile']=record;active['next_execution_not_started']=True
    active['latest_planning_status']='mixed_profile_needs_corrected_seven_arm_runtime_replay'
    active['next_substep_estimated_work_minutes']=[10,15]
    active['next_substep_timing_condition']='No new generation: separate14-update corrected replay; previous100 updates remain counted and preserved.'
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'spec_sources/downstream_profile_record_20260917.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('judgment','profile_fully_passed','generation_decodes','total_classifier_updates','corrected_complete_seven_arm_training_replay','expert_final_predictions')}))
