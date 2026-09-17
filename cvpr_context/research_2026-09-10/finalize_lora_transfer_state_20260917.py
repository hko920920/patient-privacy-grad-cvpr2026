from pathlib import Path
import json,hashlib,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CODE=ROOT.parents[1]/'code_working'
OUT=CODE/'_reports/lora_transfer_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(OUT/'result.json');verify=read(OUT/'verification.json')
    assert verify['status']=='PASS_INTEGRITY_NOT_AUTOMATIC_EFFICACY' and not result['engineering_gate_passed']
    report='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'
    record=dict(report=report,protocol='TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md',status='completed_verified_gate_failed',
        output_directory=str(OUT),runtime_implemented=True,engineering_gate_passed=False,images=256,generator_updates=896,
        classifier_updates=2400,classifier_runs=6,extra_classifier_updates=6,extra_generation_decodes=2,
        means={a:result['means'][a] for a in ['L_public','L_pooled']},comparisons=result['comparisons'],
        patient_cluster_AUROC_interval=verify['patient_cluster_intervals_95']['L_pooled-L_public']['AUROC'],verification_checks=verify['checks'],
        contract_sha256=sha(OUT/'contract.json'),result_sha256=sha(OUT/'result.json'),verification_sha256=sha(OUT/'verification.json'),
        public_record_sha256=sha(ROOT/'spec_sources/lora_transfer_execution_record_20260917.json'),
        current_full64_branch_closed=True,DP_executed=False,expert_opened=False,reserved_opened=False,final_ready=False,
        causal_failure_source_established=False,alternative_selected=False)
    next_task='Review the completed head and fixed LoRA negative development results, then decide whether one evidence-based shared downstream/data-use/measurement uncertainty warrants a separate diagnostic. No automatic adapter sweep, new algorithm, topic switch, DP, expert-final or reserved access.'
    output='A bounded next-question decision from stored evidence only; no next GPU experiment or alternative method is currently selected.'
    active_task='Existing LoRA comparison completed and verified; private incremental synthetic utility gate failed'
    opened=['Private synthetic utility remains unestablished under the tested head and fixed LoRA configurations',
        'Shared classifier, requested-label, bank and fixed-compute replacement factors are not causally separated',
        'Single LoRA trajectory/bank and reused weak-label development limit generalization',
        'No automatic DP/final/reserved or third-adapter execution; preserve both negative records']
    for name in ['research_state.json','patient_baseline_spec.json']:
        p=ROOT/name;a=read(p)
        assert 'completed_lora_transfer_comparison_20260917' not in a
        a['pre_lora_transfer_execution_current_result']=a.get('current_result_report')
        for k in ['current_result_report','last_actual_model_result_report','last_efficacy_result_report','current_decision_report']:a[k]=report
        a['completed_lora_transfer_comparison_20260917']=copy.deepcopy(record)
        a['step2_active_task']=active_task;a['next_task']=next_task;a['next_task_output']=output;a['step2_open_items']=opened
        a['current_planning_report']='TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md'
        a['active_execution'].update(status='no_running_execution',last_completed_output=str(OUT),last_completed_report=report,
            output_directory=str(OUT),additional_execution_contract_fixed=True,next_execution_not_started=True,next_execution_scheduled=False,
            new_two_track_experiment_status='existing_LoRA_comparison_completed_private_increment_gate_failed',
            latest_planning_status='frozen_LoRA_plan_executed_and_verified',stage2_completion_claim=False,
            completed_lora_transfer_comparison=copy.deepcopy(record))
        a['step2_completed_diagnostics'].append('Fixed public-only/pooled E4 LoRA continuation:896updates,256images,6classifier runs; integrity verified, private-increment gate failed; no final or DP')
        if 'steps' in a:
            for s in a['steps']:
                if s['number']==3:s.update(status='existing_LoRA_and_downstream_runtime_implemented_verified',meaning='E4 continuation, data/cache boundaries, source/kernel parity implemented and checked; no medical DP implementation added.')
                if s['number']==4:s.update(status='nonDP_head_and_fixed_LoRA_development_gates_failed',meaning='Both development studies completed; failure causes not isolated; expert final and reserved remain closed.')
        a['updated_utc']=datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Both states updated; prior completed result objects retained.')
if __name__=='__main__':main()
