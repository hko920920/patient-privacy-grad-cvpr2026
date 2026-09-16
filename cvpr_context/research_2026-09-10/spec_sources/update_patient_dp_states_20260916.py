from pathlib import Path
import json,datetime
r=Path.cwd()
t=r.parents[1]
run=t/'code_working/_reports/frozen_residual_patient_dp_20260916_v1'
a=json.loads((run/'analysis.json').read_text())
e=json.loads((run/'execution.json').read_text())
result=dict(report='TRACK1_PATIENT_DP_RESULTS_20260916.md',protocol='TRACK1_PATIENT_DP_PROTOCOL_20260916.md',
 output_directory=str(run),status='completed_independently_verified',dp_simulation=True,
 epsilon_per_fixed_mechanism=8.0,delta_per_fixed_mechanism=1e-5,adjacency='add_remove_one_patient',
 train_patients=80,development_patients=40,public_calibration_patients=32,
 models=64,noise_repeats_per_cell=16,controls=20,new_backbone_forward=0,new_backbone_backward=0,
 execution_seconds=e['total_seconds'],verification_checks=41812,verification_seconds=24.9702,
 summaries=a['summaries'],public_only_full_mse=a['controls_mean_mse']['control_full_public_only'],
 conclusions=['Compact-head denoising adaptation remains with patient DP-SGD.',
 'Current zero-centered joint SSP loses to same-head SGD and public-only ridge.',
 'Selected spectral floor is a directly controlled major observed SSP loss.',
 'Private data added value, actual generation utility, novelty and SOTA superiority remain unestablished.'])
for fn in ['research_state.json','patient_baseline_spec.json']:
 p=r/fn
 d=json.loads(p.read_text(encoding='utf-8'))
 d['pre_patient_dp_current_result_report_20260916']=d.get('current_result_report')
 d['current_result_report']=result['report']
 d['completed_track1_patient_dp_20260916']=result
 d['next_task']='Develop and compare public-reference residual protection against equally public-initialized strong DP-SGD and public-only ridge. Preserve the observed small private-data utility margin; no outcome-driven cohort or baseline weakening.'
 d['next_task_output']='Concrete fixed-objective/bias/noise derivation, closest standard public-information baselines, independent public calibration-role design, same privacy accounting and predeclared utility/cost criterion before a separate new development run.'
 d['next_task_plan_report']='spec_sources/track1_public_reference_alternatives_20260916.md'
 d['step2_active_task']='First patient-DP fixed-head comparison completed and verified; connect the observed noisy-Gram/floor loss to a public-reference design and meaningful private utility beyond public-only baselines.'
 d['current_step']=2
 if 'steps' in d:
  for s in d['steps']:
   if s['number']==3:
    s['status']='track1_patient_dp_four_cell_implementation_complete_public_reference_successor_under_design'
    s['meaning']='Fixed SSP and same-head patient-DP-SGD implemented; next public-reference candidate remains untested.'
   if s['number']==4:
    s['status']='track1_first_patient_dp_development_comparison_verified_full_paper_claim_unvalidated'
    s['meaning']='Compact adaptation survives DP-SGD; SSP superiority and private added value not established; generation/final validation pending.'
 if 'track1_dp_comparison_design_20260916' in d:
  z=d['track1_dp_comparison_design_20260916']
  z.update(status='executed_and_verified',public_calibration_extracted=True,sgd_steps_actual=2000,sgd_budget_final=True,
    final_execution_contract_frozen=True,new_dp_models_trained=64,result_report=result['report'])
 if 'active_execution' in d:
  z=d['active_execution']
  z.update(status='no_running_execution',current_step=2,last_completed_output=str(run),
      last_completed_report=result['report'],new_two_track_experiment_started=True,
      new_two_track_experiment_status='track1_patient_dp_comparison_completed_verified',
      additional_execution_contract_fixed=False,next_execution_not_started=True)
  z['completed_track1_patient_dp']=result
 if 'two_track_designs' in d and isinstance(d['two_track_designs'],dict) and isinstance(d['two_track_designs'].get('track1'),dict):
  d['two_track_designs']['track1'].update(patient_dp_report=result['report'],patient_dp_status=result['status'])
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('both authoritative states updated')

