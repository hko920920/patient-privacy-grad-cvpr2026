from pathlib import Path
import json,datetime
root=Path.cwd()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ['research_state.json','patient_baseline_spec.json']:
 p=root/name
 d=json.loads(p.read_text(encoding='utf-8'))
 d['next_task']='Extract the fixed independent public quality32 calibration statistics, validate optimizer sufficiency, and freeze clipping/scales/PSD/optimizer/seeds before the same-head patient-DP four-cell comparison.'
 d['next_task_output']='Public calibration and frozen implementation contract; public dual-RDP precheck is distinct from DP model execution; report CPU timing before running.'
 d['next_task_plan_report']='TRACK1_PATIENT_DP_COMPARISON_PLAN_20260916.md'
 d['track1_dp_comparison_design_20260916']={'status':'comparison_structure_reviewed_public_calibration_and_dp_execution_pending','report':d['next_task_plan_report'],'cells':['static16-SSP','static16-DP-SGD','full64-SSP','full64-DP-SGD'],'adjacency':'add_remove_one_patient','fixed_public_denominator':80,'epsilon':8.0,'delta':1e-5,'public_calibration_patients':32,'public_calibration_images_per_patient':2,'public_calibration_extracted':False,'sgd_q_candidate':0.1,'sgd_steps_candidate':500,'sgd_budget_final':False,'final_execution_contract_frozen':False,'new_dp_models_trained':0,'estimated_preparation_minutes':[30,60]}
 if 'step2_open_items' in d:
  old='Track1: feature API, inference-available inputs, fixed patient objective and source contract; actual nonprivate restricted-family capacity.'
  if old in d['step2_open_items']:
   d.setdefault('historical_open_items_closed_by_capacity_20260916',[]).append(old)
   d['step2_open_items'].remove(old)
 if 'active_execution' in d:
  a=d['active_execution']
  a['historical_operation_redesign_execution_state_20260916']={'new_two_track_experiment_started':a.get('new_two_track_experiment_started'),'new_two_track_experiment_status':a.get('new_two_track_experiment_status')}
  a['new_two_track_experiment_started']=True
  a['new_two_track_experiment_status']='track1_nonprivate_capacity_completed_verified_dp_comparison_not_run'
  a['latest_comparison_plan']=d['next_task_plan_report']
 d['updated_utc']=now
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('updated both states')

