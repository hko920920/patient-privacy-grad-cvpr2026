"""Record completed fixed-checkpoint diagnostic, preserving past efficacy decisions."""
from pathlib import Path
import json,hashlib,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'code_working/_reports/classifier_generalization_20260917_v1'
REPORT='TRACK1_CLASSIFIER_GENERALIZATION_RESULTS_20260917.md'
PROTOCOL='TRACK1_CLASSIFIER_GENERALIZATION_PROTOCOL_20260917.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(OUT/'result.json');v=read(OUT/'verification.json');public=read(ROOT/'spec_sources/classifier_generalization_record_20260917.json')
    assert v['status']=='PASS_FIXED_INFERENCE_DIAGNOSTIC_INTEGRITY'
    record=dict(report=REPORT,protocol=PROTOCOL,status='completed_verified_generalization_gap_observed',
       output_directory=str(OUT),new_training=0,new_generation=0,classifiers=12,
       unique_training_images=1389,new_training_image_forward=11201,development_witness_forward=768,
       model_states_unchanged=True,BN_buffers_unchanged=True,development_witness_exact=True,
       means=public['means'],contract_sha256=sha(OUT/'contract.json'),result_sha256=sha(OUT/'result.json'),
       verification_sha256=sha(OUT/'verification.json'),public_record_sha256=sha(ROOT/'spec_sources/classifier_generalization_record_20260917.json'),
       current_full64_branch_closed=True,fixed_LoRA_utility_gate_passed=False,
       method_efficacy_retested=False,causal_failure_source_established=False,
       DP_executed=False,expert_opened=False,reserved_opened=False,final_ready=False)
    before=read(OUT/'state_before.json')
    for name in ['research_state.json','patient_baseline_spec.json']:
        p=ROOT/name;a=read(p)
        assert a==before[name], 'Unexpected concurrent state modification'
        assert 'completed_classifier_generalization_20260917' not in a
        assert a['last_efficacy_result_report']=='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'
        a['pre_classifier_generalization_current_result']=a.get('current_result_report')
        for k in ['current_result_report','last_actual_model_result_report','current_decision_report']:a[k]=REPORT
        a['current_planning_report']=PROTOCOL
        a['completed_classifier_generalization_20260917']=copy.deepcopy(record)
        a['step2_active_task']='Fixed classifier inference diagnostic complete: near-perfect seen-training discrimination, poor stored development generalization; no cause isolated'
        a['next_task']='Interpret the observed train/development gap and decide whether one specific prospective data-support or classifier-recipe contrast is warranted. No automatic retraining, BN calibration, adapter sweep, DP or final/reserved access.'
        a['next_task_output']='A bounded next-comparison decision; no new experiment or solution selected by this diagnostic.'
        a['step2_open_items']=[
          'Train/eval-path-on-seen-data collapse was not observed; near-perfect seen-training discrimination coexists with weak development discrimination',
          'Limited public-positive support, memorization, cohort/weak-label differences and classifier/synthetic interactions remain causally unseparated',
          'Both previous private-synthetic utility gates remain failed; no new efficacy comparison in this diagnostic',
          'DP remains paused; expert532 and reserved4213 remain unopened']
        a['active_execution'].update(status='no_running_execution',last_completed_output=str(OUT),
           last_completed_report=REPORT,output_directory=str(OUT),additional_execution_contract_fixed=True,
           next_execution_not_started=True,next_execution_scheduled=False,
           new_two_track_experiment_status='fixed_classifier_diagnostic_complete_generalization_gap_observed',
           latest_planning_status='bounded_inference_only_diagnostic_completed',
           stage2_completion_claim=False,completed_classifier_generalization=copy.deepcopy(record))
        a['step2_completed_diagnostics'].append('Fixed12classifier checkpoint inference on actually seen training images: high training discrimination vs weak development, unchanged parameters/BN; no retraining or new efficacy')
        a['updated_utc']=datetime.now(timezone.utc).isoformat()
        for k,old in before[name].items():
            if k.startswith('completed_'):assert a[k]==old
        for k,old in before[name]['active_execution'].items():
            if k.startswith('completed_'):assert a['active_execution'][k]==old
        p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('States updated; last efficacy remains the failed fixed LoRA comparison.')
if __name__=='__main__':main()
