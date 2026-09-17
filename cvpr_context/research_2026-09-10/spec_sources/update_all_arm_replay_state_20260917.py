import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT.parents[1]/'code_working';OUT=CODE/'_reports/downstream_all_arm_replay_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OUT/'verification.json');safety=read(OUT/'entrypoint_safety.json')
if v['status']!='PASS_CORRECTED_ALL_ARM_ONE_STEP_INTEGRATION_REPLAY' or v['optimizer_updates']!=14:raise RuntimeError('Gate incomplete')
if safety['legacy_cli_exit_code']==0 or len(safety['blocked_imports'])!=2:raise RuntimeError('Entry guard incomplete')
report='TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_RESULTS_20260917.md'
sources={'report':ROOT/report,'protocol':ROOT/'TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_PROTOCOL_20260917.md',
    'contract':OUT/'contract.json','actual_result':OUT/'result.json','verification':OUT/'verification.json',
    'entrypoint_safety':OUT/'entrypoint_safety.json','logging_resume':OUT/'logging_fix_amendment.json'}
record=dict(report=report,completed_utc=datetime.now(timezone.utc).isoformat(),
    judgment='positive_corrected_all_arm_one_step_integration_not_efficacy',integration_gate_passed=True,
    optimizer_updates=14,prior_separate_profile_updates=100,old98_still_invalid=True,
    arms=7,repeats=2,steps_per_run=1,selected_source_files=61,independent_GPU_preprocessing_exact_arms=7,
    exact_final_state_replays=7,changed_trainable_parameter_tensors_each_run=62,
    actual_source_array_binding=True,legacy_classifier_CLI_blocked=True,legacy_data_train_imports_blocked=True,
    logging_resume='One fully saved update retained,13 resumed; source/data/optimizer/seed/tolerance unchanged.',
    new_generation=0,AUROC_AP_computed=False,new_DP=0,expert_pixels=0,expert_predictions=0,reserved_pixels=0,reserved_predictions=0,
    long_run_convergence_validated=False,private_utility_established=False,final_ready=False,
    full512_bank_executed=False,full21_classifier_runs_executed=False,
    next_task_eta_minutes=[90,150],remote_main_checked=False,remote_push=False,
    source_paths={k:str(p) for k,p in sources.items()},source_sha256={k:sha(p) for k,p in sources.items()})
next_task='Freeze and execute the master non-DP development package:512 fresh paired synthetic images, public R1 classifier calibration,7 arms x3seeds=21runs, corrected safe imports and actual pixel-array binding. Extend explicit full-study orchestration using corrected kernel, never legacy classifier. Update ETA from each arm first10-20steps. Evaluate S2/S3 versus BOTH S1/R1 on method-development only. Keep expert532/reserved4213 closed; no DP before non-DP gate.'
for name in ['research_state.json','patient_baseline_spec.json']:
    p=ROOT/name;s=read(p)
    if 'completed_corrected_all_arm_replay_20260917' in s:raise RuntimeError('Already recorded')
    s['pre_corrected_replay_actual_result']=s['current_result_report'];s['current_step']=2
    s['current_result_report']=s['last_actual_model_result_report']=report
    s['current_planning_report']='TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md';s['next_task_plan_report']=s['current_planning_report']
    s['next_task']=next_task;s['next_task_output']='Full non-DP method-development AUROC/AP comparison and interpretation; not expert-final or DP results.'
    s['step2_active_task']='Corrected all-arm one-step integration passed; proceed to actual non-DP downstream development utility question.'
    s['research_question_status']='Execution integration established; private incremental utility, long-run convergence, DP utility and CVPR contribution remain unestablished.'
    s['step2_open_items']=[
        'Full512-image bank/public calibration/21classifier runs and private incremental AUROC/AP remain unexecuted; use corrected code only.',
        '14updates establish all-arm one-step integration, not400/800step convergence or precise long-run throughput.',
        'Original98 wrong-source updates stay invalid; previous100-update profile and new14-update replay are separately counted.',
        'Development4053 already consumed for preprocessing/profile; expert532 and reserved4213 remain closed.',
        'Only if the non-DP development gate passes pursue new medical patient-DP, accounting, DP-LoRA and stronger augmentation/architecture/prompt robustness.',
        'Freeze all final non-DP/DP models and analysis before any expert final prediction; old192 and both evaluator failures remain unchanged.']
    s['completed_corrected_all_arm_replay_20260917']=record;s['updated_utc']=record['completed_utc']
    active=s['active_execution'];active['status']='no_running_execution';active['last_completed_report']=report;active['last_completed_output']=str(OUT)
    active['corrected_all_arm_replay']=record;active['next_execution_not_started']=True
    active['next_substep_estimated_work_minutes']=[90,150]
    active['next_substep_timing_condition']='Provisional full non-DP package estimate; update at first10-20corrected steps per arm, includes orchestration/analysis/verification.'
    active['latest_planning_status']='corrected_one_step_integration_passed_full_nonDP_utility_unexecuted'
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'spec_sources/corrected_all_arm_replay_record_20260917.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('judgment','optimizer_updates','integration_gate_passed','private_utility_established','final_ready')}))
