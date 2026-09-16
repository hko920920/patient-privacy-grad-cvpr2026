"""Update current planning pointers; preserve all completed experimental evidence."""
from pathlib import Path
import copy
import datetime
import hashlib
import json

r = Path(__file__).resolve().parents[1]
plan = 'REALISTIC_RESEARCH_PLAN_20260916.md'
prior_keys = ['next_task', 'next_task_output', 'next_task_plan_report', 'step2_active_task', 'latest_design_report', 'finalnextquestion']
shared = {
    'current_step': 2,
    'next_task': 'One bounded feasibility package: fixed patient-weight public+private non-DP heads from existing statistics, then sampling integration of existing residual heads and limited generation checks. No automatic new residual-DP variants.',
    'next_task_output': 'Pooled versus public/private-only development comparison; verified inference hook; all fixed paired generation images; measured costs; one decision report stating applicability, limits and the next utility/claim question.',
    'next_task_plan_report': plan,
    'latest_design_report': plan,
    'step2_active_task': 'Evaluate the useful medical generation-adaptation application of existing principles, separately from any optional new private-regression solver claim.',
    'finalnextquestion': 'Does this compact generation-adaptation construction merit a concrete utility-cost evaluation, with public-only and pooled controls, without assuming that known protection ingredients disqualify an application contribution?',
    'current_paper_problem_fixed': False,
    'research_question_status': 'Legitimate application candidate and feasibility question; final technical contribution and real generation utility remain unestablished.',
    'new_dp_principle_required': False,
    'new_solver_required_for_application_contribution': False,
}
replan = {
    'report': plan,
    'status': 'planning_complete_no_new_experiment',
    'user_correction': 'Known principles may produce a meaningful new application/integration contribution. Ingredient overlap is not evidence that this full application was already solved.',
    'primary_candidate': 'Useful patient-DP medical diffusion adaptation with frozen public representations and no private backbone backward, using an appropriate existing or improved protection solver.',
    'next_package_total_hours': [2, 4],
    'next_package_is_gpu_training_hours': False,
    'models_for_limited_generation': ['base', 'public_only', 'private_nonprivate', 'public_private_pooled_nonprivate', 'existing_private_DP_SGD', 'existing_private_SSP'],
    'proposed_images': 96,
    'proposed_primary_guidance': 1.0,
    'proposal_is_execution_contract': False,
    'secondary_track_auto_selected': False,
    'new_gpu_runs': 0,
    'new_training_runs': 0,
    'new_attack_runs': 0,
}
changed = []
for fn in ['research_state.json', 'patient_baseline_spec.json']:
    p = r / fn
    d = json.loads(p.read_text(encoding='utf-8'))
    completed = copy.deepcopy({k:v for k,v in d.items() if k.startswith('completed_')})
    result_report = d.get('current_result_report')
    previous = {k:d.get(k) for k in prior_keys}
    d.setdefault('pre_realistic_plan_planning_20260916', previous)
    d.update(shared)
    d['realistic_replan_20260916'] = replan
    d['step2_open_items'] = [
        'Track1: useful sampling behavior of the existing compact head; public-plus-private added value versus public-only in the fixed development setup.',
        'Track1: a concrete application/integration claim with real generation utility and total cost; use strong existing public-informed solvers if appropriate.',
        'Track1: patient-DP, public-access and objective alignment for subsequent quality-cost comparisons; formal utility evaluation is not yet ready.',
        'Track2: retained as a separate protection-evaluation question, not an automatic fallback for a weak track1 pilot.'
    ]
    for s in d.get('steps', []):
        if s['number'] == 3:
            s.update(status='fixed_head_and_first_patient_DP_implementation_complete_application_claim_under_design',
                     meaning='Generation integration and whole-application contribution require further work; new DP principles are not mandatory.')
        if s['number'] == 4:
            s.update(status='nonprivate_and_patient_DP_denoising_development_evidence_complete_generation_utility_unvalidated',
                     meaning='Existing development results are preserved; actual generated utility, private added value and final contribution are not established.')
    if isinstance(d.get('active_execution'), dict):
        d['active_execution'].update(status='no_running_execution', current_step=2,
            latest_design_review=plan, latest_planning_status='realistic_replan_complete',
            additional_execution_contract_fixed=False, next_execution_not_started=True,
            proposed_next_package_total_hours=[2,4])
    if isinstance(d.get('latest_user_clarification'), dict):
        d['latest_user_clarification'].update(current_deliverable=plan,
            known_principle_application_contribution_allowed=True,
            new_DP_principle_mandatory=False)
    if isinstance(d.get('two_track_designs'), dict):
        d['two_track_designs']['current_plan'] = plan
        d['two_track_designs']['priority'] = 'Track1 application feasibility and utility-cost contribution; standard solver reuse allowed. Track2 retained independently.'
        d['two_track_designs']['track1']['unverified'] = 'Real generation utility, private incremental value, matched total-cost advantage and contribution of the whole adaptation construction. First fixed-head patient-DP denoising comparison is complete.'
    if fn == 'patient_baseline_spec.json':
        d['historical_attack_spec_scope_notice'] = 'problem_fixed, primary_access, scenarios and attack baseline fields describe the preserved earlier patient-MIA specification; they do not fix the current protection-learning paper claim.'
    d['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert completed == {k:v for k,v in d.items() if k.startswith('completed_')}
    assert d.get('current_result_report') == result_report
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    changed.append(fn)
print(json.dumps({'updated':changed, 'completed_experimental_records_unchanged':True, 'plan_sha256':hashlib.sha256((r/plan).read_bytes()).hexdigest()}))
