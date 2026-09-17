"""Register prospective plan only; preserve diagnostic and efficacy result pointers."""
from pathlib import Path
import json,hashlib,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'code_working/_reports/real_support_plan_20260917_v1'
PROTOCOL='TRACK1_REAL_SUPPORT_DIAGNOSTIC_PROTOCOL_20260917.md'
REVIEW='TRACK1_REAL_SUPPORT_DIAGNOSTIC_PLAN_REVIEW_20260917.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(OUT/'plan.json');v=read(OUT/'verification.json');before=read(OUT/'state_before.json')
    assert v['status']=='PASS_METADATA_ROLE_BOUNDARIES_AND_INDEPENDENT_SCHEDULE'
    record=dict(protocol=PROTOCOL,review=REVIEW,status='metadata_verified_runtime_not_implemented',
       output_directory=str(OUT),plan_sha256=sha(OUT/'plan.json'),verification_sha256=sha(OUT/'verification.json'),
       public_plan='spec_sources/real_support_diagnostic_plan_20260917.json',
       counts=plan['counts'],planned_exposure=plan['planned_exposure'],
       planned_main_updates=1200,planned_probe_updates=4,planned_probe_seed=11,
       new_model_execution=0,new_training=0,new_generation=0,new_pixel_decode=0,
       original_roles_changed=False,role_overlay_status='planned_not_yet_optimizer_consumed',
       runtime_implemented=False,DP_executed=False,expert_opened=False,reserved_opened=False,final_ready=False,
       improved_performance_established=False,cause_established=False)
    for name,old in before.items():
        p=ROOT/name;a=read(p)
        assert a==old,'Unexpected state changes'
        assert 'completed_real_support_plan_20260917'not in a
        a['completed_real_support_plan_20260917']=copy.deepcopy(record)
        # No new inference or efficacy: all three actual result pointers stay unchanged.
        assert a['last_actual_model_result_report']=='TRACK1_CLASSIFIER_GENERALIZATION_RESULTS_20260917.md'
        assert a['last_efficacy_result_report']=='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'
        a['current_planning_report']=PROTOCOL;a['current_decision_report']=REVIEW
        a['step2_active_task']='One fixed-compute expanded real-patient pool comparison designed; metadata schedule verified, new runtime/model execution pending'
        a['next_task']='Implement the Rwide diagnostic pool provider with legacy guards and unchanged classifier kernels; freeze runtime sources and pass four seed11 one-step parity/replay updates before three fixed400-step runs. Preserve prior results and locked cohorts; no automatic DP.'
        a['next_task_output']='A verified runtime followed by the single predeclared Rwide-R1 development comparison; no classifier or pool sweep.'
        a['step2_open_items']=[
          'Current real-only and synthetic paths both show a train/development generalization gap',
          'Rwide pool5910/2699 with120positive patients is frozen as a diagnostic NONDP expansion, not a public baseline',
          'Actual expanded-pool performance and runtime source/pixel/kernel parity are not yet tested',
          'Former selection2027 must not be reused as held-out validation after its future training use',
          'Prior head/LoRA gates remain failed; expert532/reserved4213 and DP boundaries preserved']
        a['active_execution'].update(status='no_running_execution',latest_design_review=REVIEW,
           latest_planning_status='real_support_plan_metadata_verified_no_new_model',
           next_execution_not_started=True,next_execution_scheduled=False,
           completed_real_support_plan=copy.deepcopy(record))
        a['step2_completed_diagnostics'].append('Metadata-only prospective Rwide real-support comparison:2699patients5910images120positivepatients; original R1 sampler and38400 planned slots checked; no new model/pixel execution')
        a['updated_utc']=datetime.now(timezone.utc).isoformat()
        for k in ['current_result_report','last_actual_model_result_report','last_efficacy_result_report']:
            assert a[k]==old[k]
        for k,val in old.items():
            if k.startswith('completed_'):assert a[k]==val
        p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Plan registered; current actual and efficacy pointers preserved.')
if __name__=='__main__':main()
