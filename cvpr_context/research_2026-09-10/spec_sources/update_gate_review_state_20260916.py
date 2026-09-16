"""Record corrected gates and verified evaluator readiness without changing experiments."""
from pathlib import Path
import copy, datetime, json
r=Path(__file__).resolve().parents[1]
review={
    'status':'completed_source_and_readiness_review_no_new_experiment',
    'report':'REALISTIC_RESEARCH_PLAN_20260916.md',
    'source_note':'spec_sources/public_assisted_deep_priors_20260916.md',
    'gate_note':'spec_sources/research_gate_review_20260916.md',
    'readiness_note':'spec_sources/generation_evaluator_readiness_20260916.md',
    'generation_metrics_implemented_and_preflight_verified':True,
    'original_real_reference_patients':448,
    'overlap_with_current_roles':{'fit':32,'development':22,'calibration':59,'test':65,'public32':11},
    'eligible_reference_candidates_after_patient_exclusions':259,
    'candidate_reference_is_fresh_independent_final_test':False,
    'old_per_image_encoder_features_saved':False,
    'prompt_reference_mixture_requires_new_specification':True,
    'next_package_total_work_hours':[2.5,5],
    'exploratory_images_per_method':16,
    'initial_DP_head_selection':'fixed manifest index0 per method; no best-model selection or pooled different-head generator metrics',
    'KID_subset50_requires_at_least50_per_method_but_not_power_guarantee':True,
    'parameter_residual_stability_required':False,
    'reference_models_are_generation_utility_upper_bounds':False,
    'strict_cost_superiority_to_public_only_required':False,
    'novel_DP_solver_required':False,
    'new_experiments':0,
}
for fn in ['research_state.json','patient_baseline_spec.json']:
    p=r/fn
    d=json.loads(p.read_text(encoding='utf-8'))
    old=copy.deepcopy({k:v for k,v in d.items() if k.startswith('completed_')})
    prior_result=d.get('current_result_report')
    d.setdefault('pre_gate_review_realistic_replan_20260916',copy.deepcopy(d.get('realistic_replan_20260916')))
    d['followup_gate_review_20260916']=review
    d['next_task']='One bounded pooled-control, generation-integration and exploratory-metric package using existing evaluator code, patient-disjoint reference candidates and matched prompt mixtures.'
    d['next_task_output']='Fixed pooled references, verified sampling hook, all paired images, eligible reference manifest and exploratory RAD-DINO/BioViL-T scores with sample-size limits, measured costs and a decision report.'
    d['next_task_plan_report']='REALISTIC_RESEARCH_PLAN_20260916.md'
    d['next_package_total_work_hours']=[2.5,5]
    if isinstance(d.get('realistic_replan_20260916'),dict):
        d['realistic_replan_20260916'].update(next_package_total_hours=[2.5,5],
            evaluator_followup_review=review['readiness_note'],
            proposed_DP_head_policy=review['initial_DP_head_selection'])
    if isinstance(d.get('active_execution'),dict):
        d['active_execution'].update(status='no_running_execution',
            latest_planning_status='followup_gates_and_evaluator_readiness_review_complete',
            proposed_next_package_total_hours=[2.5,5],additional_execution_contract_fixed=False,
            next_execution_not_started=True)
    d['current_research_questions']=[
        'Does protected use of private patient data add meaningful generation utility over the available public-only adaptation?',
        'Does this generation-adaptation construction improve the utility-total-cost tradeoff against strong patient-DP learners at matched public access and privacy constraints?'
    ]
    d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert old=={k:v for k,v in d.items() if k.startswith('completed_')}
    assert prior_result==d.get('current_result_report')
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

# Only update the current navigation panel; historical estimates remain history.
p=r/'build_review.py'
s=p.read_text(encoding='utf-8')
old='구현·검토·분석 포함 총2–4시간의 제한된 작업입니다.'
new='기존 RAD-DINO/BioViL-T 평가기와 환자 분리 참조를 연결하며 구현·검토·분석 포함 총2.5–5시간으로 조정했습니다.'
assert s.count(old)==1
p.write_text(s.replace(old,new),encoding='utf-8')
print('states and current navigation updated; completed evidence unchanged')
